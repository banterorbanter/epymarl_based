import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import numpy as np
from torch.optim import RMSprop, AdamW
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets


# learning for 6h_vs_8z scenario
class QLearner_7:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())

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

        self.params += list(self.mac.msg_rnn.parameters())
        # self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser = AdamW(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.loss_weight = [0.5, 1, 1.5]  # this is the beta in the Algorithm 1

        self.n_agents = args.n_agents
        self.n_action = args.n_actions
        self.dummy_buffer = [dict() for _ in range(args.n_agents)]
        self.target_dummy_buffer = [dict() for _ in range(args.n_agents)]

        self.train_t = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        ################ store the previous mac################
        previous_msg_list = []
        smooth_loss_list = []
        regulariation_smooth = 3.0
        regulariation_robust = 0.08
        # regulariation_smooth = 300
        # regulariation_robust = 8

        # th.autograd.set_detect_anomaly(True)

        self.mac.init_hidden(batch.batch_size)
        bs = batch.batch_size
        smooth_loss = th.zeros((bs * self.n_agents)).cuda()
        for t in range(batch.max_seq_length):
            agent_local_outputs, input_hidden_states, vi = self.mac.forward(batch, t=t)
            input_hidden_states = input_hidden_states.view(-1, 64)
            self.mac.hidden_states_msg, dummy = self.mac.msg_rnn(self.mac.hidden_states_msg, input_hidden_states)
            ss = min(len(previous_msg_list), 3)
            # compute the l2 distance in the window
            for i in range(ss):
                ll = (((dummy - previous_msg_list[i]) ** 2).sum(dim=1))
                smooth_loss = smooth_loss + self.loss_weight[i] * ll / (
                    (ss * bs * self.n_agents * self.n_action * (dummy ** 2)).sum(dim=1))
            previous_msg_list.append(dummy.cuda())
            if (len(previous_msg_list) > 3): previous_msg_list.pop(0)
            smooth_loss_reshape = smooth_loss.reshape(bs, self.n_agents, 1).sum(1)  # (32,1)
            smooth_loss_list.append(smooth_loss_reshape)
            # generate the message
            # dummy_final = dummy.reshape(bs,7,8).detach().clone()
            dummy_final = dummy.reshape(bs, self.n_agents, self.n_action).clone()
            d_dummy = dummy_final
            # if self.args.msg_delay_type == 'N_distribution' and test_mode:
            #     latency = th.normal(mean=self.args.delay_value, std=self.args.delay_scale, size=(1, self.n_agents))
            # elif self.args.msg_delay_type == 'constant' and test_mode:
            #     latency = th.full((1, self.args.n_agents), self.args.delay_value)
            # else:
            #     latency = th.zeros((1, self.args.n_agents))
            # latency = th.max(th.tensor(0), latency).int()

            # for i in range(self.n_agents):
            #     self.dummy_buffer[i][t + latency[0][i].item()] = dummy_final[:, i, :]

            # d_dummy = th.zeros_like(dummy_final).cuda()
            # for i in range(self.n_agents):
            #     if t in self.dummy_buffer[i]:
            #         d_dummy[:, i, :] = self.dummy_buffer[i].pop(t)

            d_dummys = th.zeros((bs, self.n_action)).cuda()
            for i in range(self.n_agents):
                d_dummys = d_dummys + d_dummy[:, i, :].cuda()

            agent_global_outputs = th.zeros((bs, self.n_agents, self.n_action)).cuda()
            for i in range(self.n_agents):
                agent_global_outputs[:, i, :] = ((d_dummys - d_dummy[:, i, :]) / float(self.n_agents - 1)).view(bs,
                                                                                                                self.n_action).cuda()

            agent_outs = agent_local_outputs + agent_global_outputs.cuda()
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time   #(32,T,7,8)
        ############compute the robustness loss##################
        robust_loss = th.topk(mac_out, 2)[0][:, :, :, 0] - th.topk(mac_out, 2)[0][:, :, :, 1]
        robust_loss = th.exp(-25.0 * robust_loss).sum(dim=2)[:, :-1].unsqueeze(2) / (
                bs * (self.n_agents - 1))  # (32,38)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []

            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_local_outputs, target_input_hidden_states, tvi = self.target_mac.forward(batch, t=t)
                target_input_hidden_states = target_input_hidden_states.view(-1, 64)
                self.target_mac.hidden_states_msg, target_dummy = self.target_mac.msg_rnn(
                    self.target_mac.hidden_states_msg, target_input_hidden_states)

                target_dummy = target_dummy.reshape(bs, self.n_agents, self.n_action).detach().clone()
                d_target_dummy = target_dummy
                # if self.args.msg_delay_type == 'N_distribution' and test_mode:
                #     latency = th.normal(mean=self.args.delay_value, std=self.args.delay_scale, size=(1, self.n_agents))
                # elif self.args.msg_delay_type == 'constant' and test_mode:
                #     latency = th.full((1, self.args.n_agents), self.args.delay_value)
                # else:
                #     latency = th.zeros((1, self.args.n_agents))
                # latency = th.max(th.tensor(0), latency).int()

                # for i in range(self.n_agents):
                #     self.target_dummy_buffer[i][t + latency[0][i].item()] = target_dummy[:, i, :]

                # d_target_dummy = th.zeros_like(target_dummy).cuda()
                # for i in range(self.n_agents):
                #     if t in self.dummy_buffer[i]:
                #         d_target_dummy[:, i, :] = self.target_dummy_buffer[i].pop(t)

                d_target_dummys = th.zeros((bs, self.n_action)).cuda()
                for i in range(self.n_agents):
                    d_target_dummys = d_target_dummys + d_target_dummy[:, i, :].cuda()

                target_agent_global_outputs = th.zeros((bs, self.n_agents, self.n_action)).cuda()
                for i in range(self.n_agents):
                    target_agent_global_outputs[:, i, :] = (
                            (d_target_dummys - d_target_dummy[:, i, :]) / float(self.n_agents - 1)).view(bs,
                                                                                                         self.n_action).cuda()

                target_agent_outs = target_agent_local_outputs + target_agent_global_outputs.cuda()
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
            
            if self.mixer is not None:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

        # Mix
        if self.mixer is not None:
            # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

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
        td_error = (chosen_action_qvals - targets.detach())  # (32,25,1)
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        ######compute the smooth_loss and robust_loss#########
        smooth_loss = th.stack(smooth_loss_list[0:-1], dim=1)
        smooth_loss = (smooth_loss * mask).sum() / mask.sum() * regulariation_smooth
        robust_loss = (robust_loss * mask).sum() / mask.sum() * regulariation_robust

        loss = (
                       masked_td_error ** 2).sum() / mask.sum() + smooth_loss + robust_loss
        # loss = (masked_td_error ** 2).sum() / mask.sum()
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.train_t += 1
        # if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
        #     self._update_targets()
        #     self.last_target_update_episode = episode_num
        if (self.train_t - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = self.train_t

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

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
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
