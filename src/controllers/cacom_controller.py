from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from modules.agents.cacom_delay_agent import ExpGate


# This multi-agent controller shares parameters between agents
class CACOM_MAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.gate = ExpGate(args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # Different API!!!!!
        agent_outputs, _, _ = self.forward(
            ep_batch, t_ep, test_mode=test_mode, train_mode=False
        )
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, all_through=False, test_mode=False, **kwargs):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, losses, reply_freq = self.agent.forward(
            agent_inputs,
            self.hidden_states,
            ep_batch.batch_size,
            self.gate,
            t,
            all_through=all_through,
            test_mode=test_mode,
            **kwargs
        )

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True
                    ).float()

                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return (
            agent_outs.view(ep_batch.batch_size, self.n_agents, -1),
            losses,
            reply_freq,
        )

    def forward_gate(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        # (bs, n_agents, n_actions)
        avail_actions = ep_batch["avail_actions"][:, t]
        self.hidden_states, probs, q_pos, q_neg, idx = self.agent.cal_gate_labels(
            agent_inputs, self.hidden_states, ep_batch.batch_size, self.gate, t
        )

        q_pos = q_pos[:, [i for i in range(self.n_agents) if i != idx], :].reshape(
            ep_batch.batch_size * (self.n_agents - 1), -1
        )
        q_neg = q_neg[:, [i for i in range(self.n_agents) if i != idx], :].reshape(
            ep_batch.batch_size * (self.n_agents - 1), -1
        )

        # TODO: remove ego
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                avail_actions = avail_actions[
                    :, [i for i in range(self.n_agents) if i != idx], :
                ]
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * (self.n_agents - 1), -1
                )
                q_pos[reshaped_avail_actions == 0] = -1e10
                q_neg[reshaped_avail_actions == 0] = -1e10

            q_pos = th.nn.functional.softmax(q_pos, dim=-1)
            q_neg = th.nn.functional.softmax(q_neg, dim=-1)

        q_pos = th.max(q_pos, dim=-1)[0]
        q_neg = th.max(q_neg, dim=-1)[0]
        q_diff = q_pos - q_neg

        return q_diff, probs.reshape((ep_batch.batch_size * (self.n_agents - 1), 2))

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def gate_parameters(self):
        return self.gate.parameters()

    def cuda(self):
        self.agent.cuda()
        self.gate.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.gate.state_dict(), "{}/gate.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.gate.load_state_dict(
            th.load(
                "{}/gate.th".format(path), map_location=lambda storage, loc: storage
            )
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
