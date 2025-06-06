import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, AdamW
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets


class CACOM_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.gate_params = list(mac.gate_parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # self.optimiser = RMSprop(
        #     params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        # )
        self.optimiser = AdamW(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 1e-6))

        # self.gate_optimizer = RMSprop(
        #     params=self.gate_params,
        #     lr=args.gate_lr,
        #     alpha=args.optim_alpha,
        #     eps=args.optim_eps,
        # )
        self.gate_optimizer = AdamW(params=self.gate_params,  lr=args.gate_lr, weight_decay=getattr(args, "weight_decay", 1e-6))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_gate_t = -self.args.train_gate_intervel - 1

        self.train_t = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        all_through = t_env < self.args.start_train_gate

        # NOTE: record logging signal
        prepare_for_logging = (
            True
            if t_env - self.log_stats_t >= self.args.learner_log_interval
            else False
        )

        logs = []
        losses = []
        reply_freqs = []

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs, returns_, reply_freq = self.mac.forward(
                batch,
                t=t,
                prepare_for_logging=prepare_for_logging,
                train_mode=True,
                all_through=all_through,
                mixer=self.target_mixer,
            )
            mac_out.append(agent_outs)
            if prepare_for_logging and "logs" in returns_:
                logs.append(returns_["logs"])
                del returns_["logs"]
            losses.append(returns_)
            reply_freqs.append(reply_freq)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1]
            )
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"]
            )

        # Calculate 1-step Q-Learning targets
        if self.args.env != 'melting_pot' and self.args.env != 'meltingpot':
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals[:, 1:]
        else:
            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        external_loss, loss_dict = self._process_loss(losses, batch)
        loss += external_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # if (
        #     episode_num - self.last_target_update_episode
        # ) / self.args.target_update_interval >= 1.0:
        #     self._update_targets()
        #     self.last_target_update_episode = episode_num
        self.train_t += 1
        if (self.train_t - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = self.train_t

        # train gate
        if t_env - self.train_gate_t >= self.args.train_gate_intervel:
            self.mac.init_hidden(batch.batch_size)
            probs = []
            q_diffs = []

            for t in range(batch.max_seq_length):
                q_diff, prob = self.mac.forward_gate(batch, t=t)
                q_diffs.append(q_diff)
                probs.append(prob)

            q_diffs = th.cat(q_diffs, dim=0)
            probs = th.cat(probs, dim=0)
            if all_through:
                labels = th.zeros_like(q_diffs).long()
            else:
                labels = (q_diffs <= self.args.cut_off_threshold).long()

            gate_loss = F.cross_entropy(probs, labels)

            self.gate_optimizer.zero_grad()
            gate_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.gate_params, self.args.grad_norm_clip
            )
            self.gate_optimizer.step()

            self.train_gate_t = t_env

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            reply_freqs = th.stack(reply_freqs, dim=0).mean()
            self.logger.log_stat("reply_freqs", reply_freqs.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if t_env - self.train_gate_t >= self.args.train_gate_intervel:
                self.logger.log_stat("gate_loss", gate_loss.item(), t_env)
                self.logger.log_stat("q_diff_mean", q_diffs.mean(), t_env)
                self.logger.log_stat("q_diff_max", q_diffs.max(), t_env)
                self.logger.log_stat("q_diff_min", q_diffs.min(), t_env)
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

            self._log_for_loss(loss_dict, t_env)

            self.log_stats_t = t_env

    def _process_loss(self, losses: list, batch: EpisodeBatch):
        total_loss = 0
        loss_dict = {}
        for item in losses:
            for k, v in item.items():
                if str(k).endswith("loss"):
                    loss_dict[k] = loss_dict.get(k, 0) + v
                    total_loss += v
        for k in loss_dict.keys():
            loss_dict[k] /= batch.max_seq_length
        total_loss /= batch.max_seq_length
        return total_loss, loss_dict

    def _log_for_loss(self, losses: dict, t):
        for k, v in losses.items():
            self.logger.log_stat(k, v.item(), t)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
