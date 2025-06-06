import copy

import torch

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from learners.learner import Learner

from utils.maker import MixerMaker
from utils.th_utils import get_parameters_num
from utils.rl_utils import build_q_lambda_targets, build_td_lambda_targets, new_build_td_lambda_targets


class QLearner(Learner):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger
        self.device = args.device

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        if args.mixer is not None:
            self.mixer = MixerMaker.make(args.mixer, args)
        else:
            self.mixer = torch.nn.Identity()

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        logger.info(f"Mixer Size: {get_parameters_num(self.mixer.parameters())}")

        match getattr(self.args, "optimiser", "rms").lower():
            case "adam":
                self.optimiser = torch.optim.Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
            case "adamw":
                self.optimiser = torch.optim.AdamW(params=self.params, lr=args.lr)
            case "sgd":
                self.optimiser = torch.optim.SGD(params=self.params, lr=args.lr)
            case _:
                self.optimiser = torch.optim.RMSprop(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])    # Mask the final step.
        avail_actions: torch.Tensor = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            # rewards = rewards.expand(-1, -1, self.n_agents)

        # Calculate estimated Q-Values
        self.mac.agent.train()
        self.mac.init_hidden(batch.batch_size)
        t = slice(0, batch.max_seq_length)
        mac_out = self.mac.forward(batch, t=t)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = torch.gather(mac_out, dim=3, index=actions).squeeze(3)  # Remove the last dim
        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        # Calculate the Q-Values necessary for the target
        with torch.no_grad():
            self.target_mac.agent.train()

            # mac forward.
            self.target_mac.init_hidden(batch.batch_size)
            t = slice(0, batch.max_seq_length)
            target_mac_out = self.target_mac.forward(batch, t=t)

            # Mask out unavailable actions
            target_mac_out = torch.masked_fill(target_mac_out, avail_actions == 0, -1e7)

            # Max over target Q-Values
            if self.args.double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.detach().clone()
                mac_out_detach = torch.masked_fill(mac_out_detach, avail_actions==0, -1e7)
                cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
                target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if self.args.standardise_returns is True:
                target_max_qvals = (
                    target_max_qvals * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean
                )

            match target_type := getattr(self.args, "target_type", "td"):
                case "td":
                    targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
                case "td_lambda":
                    targets = new_build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                      self.args.gamma, self.args.td_lambda)
                case "q_lambda":
                    qvals = torch.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                    qvals = self.target_mixer(qvals, batch["state"])
                    targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                     self.args.gamma, self.args.td_lambda)
                case _:
                    raise ValueError(f"Invalid target type {target_type}")

            if self.args.standardise_returns:
                self.ret_ms.update(targets)
                targets = (targets - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.to(self.device)
        self.target_mac.to(self.device)
        if self.mixer is not None:
            self.mixer.to(self.device)
            self.target_mixer.to(self.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
