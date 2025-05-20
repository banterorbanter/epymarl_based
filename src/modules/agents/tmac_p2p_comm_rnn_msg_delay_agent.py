import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


class RnnMsgAgent(nn.Module):
    """
    input shape: [batch_size, in_feature]
    output shape: [batch_size, n_actions]
    hidden state shape: [batch_size, hidden_dim]
    """

    def __init__(self, input_dim, args):
        super().__init__()

        self.args = args
        self.fc1 = nn.Linear(input_dim, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        self.fc_value = nn.Linear(args.hidden_dim, args.n_value)
        self.fc_key = nn.Linear(args.hidden_dim, args.n_key)
        self.fc_query = nn.Linear(args.hidden_dim, args.n_query)

        self.fc_attn = nn.Linear(args.n_query + args.n_key * args.n_agents, args.n_agents)

        self.fc_attn_combine = nn.Linear(args.n_value + args.hidden_dim, args.hidden_dim)

        self.fc_msg_1 = nn.Linear(input_dim, args.hidden_dim)
        self.fc_msg_2 = nn.Linear(args.hidden_dim, args.n_agents)

        self.evidence_buffer = [dict() for _ in range(args.n_agents)]
        self.target_buffer = [dict() for _ in range(args.n_agents)]

    def forward(self, x, hidden):
        """
        hidden state: [batch_size, n_agents, hidden_dim]
        q_without_communication
        """
        x = F.relu(self.fc1(x))
        h_in = hidden.view(-1, self.args.hidden_dim)
        h_out = self.rnn(x, h_in)
        h_out = h_out.view(-1, self.args.n_agents, self.args.hidden_dim)
        return h_out

    def generate_send_prob(self, obs):
        batch = int(obs.shape[0] / self.args.n_agents)
        f = self.fc_msg_1(obs)
        s = self.fc_msg_2(f)
        s = nn.Softmax(dim=-1)(s).reshape(batch, self.args.n_agents, s.shape[-1])
        return s

    def q_without_communication(self, h_out):
        q_without_comm = self.fc2(h_out)
        return q_without_comm

    def communicate(self, hidden):
        """
        input: hidden [batch_size, n_agents, hidden_dim]
        output: key, value, signature
        """
        key = self.fc_key(hidden)
        value = self.fc_value(hidden)
        query = self.fc_query(hidden)

        return key, value, query

    def aggregate(self, query, key, value, hidden, send_target, t, test_mode=False):
        """
        query: [batch_size, n_agents, n_query]
        key: [batch_size, n_agents, n_key]
        value: [batch_size, n_agents, n_value]
        """
        n_agents = self.args.n_agents
        _key = torch.cat([key[:, i, :] for i in range(n_agents)], dim=-1).unsqueeze(1).repeat(1, n_agents, 1)
        query_key = torch.cat([query, _key], dim=-1)  # [batch_size, n_agents, n_query + n_agents*n_key]

        # attention weights
        attn_weights = F.softmax(self.fc_attn(query_key), dim=-1)  # [batch_size, n_agents, n_agents]

        # attentional value
        attn_applied = torch.bmm(attn_weights, value)  # [batch_size, n_agents, n_value]

        # shortcut connection: combine with agent's own hidden
        attn_combined = torch.cat([attn_applied, hidden], dim=-1)

        # used when ablate 'shortcut' connection
        # attn_combined = attn_applied

        attn_combined = F.relu(self.fc_attn_combine(attn_combined))

        # mlp, output Q
        q = self.fc2(attn_combined)  # [batch_size, n_agents, n_actions]
        evidence = torch.clamp(q, 0, torch.inf)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        ori_u = self.args.n_actions / S
        received_evidence = self.combine_message(evidence, send_target, t, test_mode)
        evidence = evidence + received_evidence
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        cmb_u = self.args.n_actions / S
        # print(evidence.shape)
        return evidence, (cmb_u - ori_u).detach()

    @torch.no_grad()
    # def combine_message(self, evidence, send_target):
    #     batch_send_target = send_target.transpose(1, 2)
    #     batch_evidence = evidence.clone().detach()
    #     alpha = batch_evidence + 1
    #     S = torch.sum(alpha, dim=-1, keepdim=True)
    #     batch_belief = (batch_evidence / (S.expand(batch_evidence.shape))).transpose(0, 1)
    #     batch_uncertainty = (self.args.n_actions / S).transpose(0, 1)
    #     received_belief = torch.zeros((batch_evidence.shape[0], self.args.n_actions)).cuda()
    #     received_uncertainty = torch.ones((batch_evidence.shape[0], 1)).cuda()
    #     for b, u in zip(batch_belief, batch_uncertainty):
    #         received_belief, received_uncertainty = self.combine(received_belief, received_uncertainty, b, u)
    #     all_evidence = self.belief_to_evidence(received_belief, received_uncertainty)
    #
    #     received_evidence = torch.tensor([]).cuda()
    #     for env_send_target, env_evidence, env_all_evi in zip(batch_send_target, batch_evidence, all_evidence):
    #         for agent_send_target in env_send_target:
    #             r_e = env_all_evi.clone()
    #             for target, e in zip(agent_send_target, env_evidence):
    #                 if target.item() == 0:
    #                     r_e = r_e - e / self.args.n_agents
    #             received_evidence = torch.cat((received_evidence, r_e.unsqueeze(0)), dim=0)
    #     received_evidence = received_evidence.reshape(evidence.shape[0], self.args.n_agents,
    #                                                   received_evidence.shape[-1])
    #
    #     scale = (torch.max(batch_evidence) - torch.min(batch_evidence)) * self.args.comm_coef
    #     received_evidence = scale * (received_evidence - torch.min(received_evidence)) / (torch.max(
    #         received_evidence) - torch.min(received_evidence) + 0.01)
    #     # print(batch_evidence.shape, received_evidence.shape)
    #     combined_evidence = batch_evidence + received_evidence
    #     # print(combined_evidence.shape)
    #     return combined_evidence.float()

    # @torch.no_grad()
    # def combine_message(self, evidence, send_target):
    #     batch_evidence = evidence.detach()
    #     batch_send_target = send_target.transpose(-2, -1)
    #     alpha = batch_evidence + 1
    #     S = torch.sum(alpha, dim=-1, keepdim=True)
    #     batch_belief = batch_evidence / (S.expand(batch_evidence.shape))
    #     batch_uncertainty = self.args.n_actions / S
    #     received_belief = torch.zeros((1, self.args.n_actions)).cuda()
    #     received_uncertainty = torch.ones((1, 1)).cuda()
    #     for env_belief, env_uncertainty in zip(batch_belief, batch_uncertainty):
    #         for b, u in zip(env_belief, env_uncertainty):
    #             received_belief, received_uncertainty = self.combine(received_belief, received_uncertainty, b, u)
    #     all_evidence = self.belief_to_evidence(received_belief, received_uncertainty)
    #
    #     received_evidence = torch.tensor([]).cuda()
    #     for env_send_target, env_evidence in zip(batch_send_target, batch_evidence):
    #         for agent_send_target in env_send_target:
    #             r_e = all_evidence.clone()
    #             for target, e in zip(agent_send_target, env_evidence):
    #                 if target.item() == 0:
    #                     r_e = r_e - e/self.args.n_agents
    #             received_evidence = torch.cat((received_evidence, r_e), dim=0)
    #     received_evidence = received_evidence.reshape(evidence.shape[0], self.args.n_agents, received_evidence.shape[-1])
    #     scale = torch.max(evidence) * self.args.comm_coef
    #     received_evidence = scale*(received_evidence-torch.min(received_evidence))/torch.max(received_evidence)-torch.min(received_evidence)
    #     combined_evidence = evidence + received_evidence
    #     return combined_evidence

    @torch.no_grad()
    def combine_message(self, evidence, send_target, t, test_mode=False):

        if self.args.msg_delay_type == 'N_distribution' and test_mode:
            latency = torch.normal(mean=self.args.delay_value, std=self.args.delay_scale, size=(1, self.args.n_agents))
        elif self.args.msg_delay_type == 'constant' and test_mode:
            latency = torch.full((1, self.args.n_agents), self.args.delay_value)
        else:
            latency = torch.zeros((1, self.args.n_agents))
        latency = torch.max(torch.tensor(0), latency).int()

        for i in range(self.args.n_agents):
            self.evidence_buffer[i][t + latency[0][i].item()] = evidence[:, i, :]
            self.target_buffer[i][t + latency[0][i].item()] = send_target[:, i, :]

        batch_size, n_agents, n_actions = evidence.shape
        d_evidence = torch.zeros_like(evidence)
        d_send_target = torch.zeros_like(send_target)
        for i in range(self.args.n_agents):
            if t in self.evidence_buffer[i]:
                d_evidence[:, i, :] = self.evidence_buffer[i].pop(t)
                d_send_target[:, i, :] = self.target_buffer[i].pop(t)

        # bs * n_agent * n_actions
        batch_evidence = evidence.detach()
        batch_send_target = send_target.transpose(-2, -1)
        alpha = batch_evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        batch_belief = batch_evidence / (S.expand(batch_evidence.shape))
        batch_uncertainty = self.args.n_actions / S

        d_batch_evidence = d_evidence.detach()
        d_batch_send_target = d_send_target.transpose(-2, -1)
        d_alpha = batch_evidence + 1
        d_S = torch.sum(d_alpha, dim=-1, keepdim=True)
        d_batch_belief = d_batch_evidence / (d_S.expand(d_batch_evidence.shape))
        d_batch_uncertainty = self.args.n_actions / d_S

        # received_belief = torch.zeros((batch_size, 1, self.args.n_actions)).cuda()
        # received_uncertainty = torch.ones((batch_size, 1, 1)).cuda()
        # combined_belief = torch.tensor([]).cuda()
        # combined_uncertainty = torch.tensor([]).cuda()
        received_belief = torch.zeros((batch_size, 1, self.args.n_actions))
        received_uncertainty = torch.ones((batch_size, 1, 1))
        combined_belief = torch.tensor([])
        combined_uncertainty = torch.tensor([])
        count = 0
        for j in range(n_agents):
            r_b = received_belief.clone()
            r_u = received_uncertainty.clone()
            dbb = torch.where(d_send_target[:, j, :].unsqueeze(2).expand((batch_size, n_agents, n_actions)).bool(),
                             d_batch_belief, 0)
            dbu = torch.where(d_send_target[:, j, :].unsqueeze(2).bool(),
                              d_batch_uncertainty, 1)
            for h in range(n_agents):
                if h == j:
                    r_b, r_u = self.combine(r_b, r_u, batch_belief[:, h, :].unsqueeze(1), batch_uncertainty[:, h, :].unsqueeze(1))
                else:
                    r_b, r_u = self.combine(r_b, r_u, dbb[:, h, :].unsqueeze(1), dbu[:, h, :].unsqueeze(1))
            combined_belief = torch.cat((combined_belief, r_b), dim=1)
            combined_uncertainty = torch.cat((combined_uncertainty, r_u), dim=1)

        # a_received_belief = torch.zeros((1, self.args.n_actions))
        # a_received_uncertainty = torch.ones((1, 1))
        # a_combined_belief = torch.tensor([])
        # a_combined_uncertainty = torch.tensor([])
        # for i, (env_send_target, env_belief, env_uncertainty) in enumerate(zip(d_batch_send_target, batch_belief, batch_uncertainty)):
        #     for j, agent_send_target in enumerate(env_send_target):
        #         r_b = a_received_belief.clone()
        #         r_u = a_received_uncertainty.clone()
        #         for h, (target, b, u) in enumerate(zip(agent_send_target, env_belief, env_uncertainty)):
        #             if target.item() == 1 or h == j:
        #                 count += 1
        #                 if h == j:
        #                     r_b, r_u = self.combine(r_b, r_u, b.unsqueeze(0), u.unsqueeze(0), tt=True)
        #                 else:
        #                     r_b, r_u = self.combine(r_b, r_u, d_batch_belief[i][h].unsqueeze(0), d_batch_uncertainty[i][h].unsqueeze(0), tt=True)
        #         a_combined_belief = torch.cat((a_combined_belief, r_b), dim=0)
        #         a_combined_uncertainty = torch.cat((a_combined_uncertainty, r_u), dim=0)
        # if not torch.equal(a_combined_belief.view(-1, 20, 2), combined_belief) or \
        #     not torch.equal(a_combined_uncertainty.view(-1, 20, 1), combined_uncertainty):
        #     print("??")
        combined_belief = combined_belief.reshape(evidence.shape[0], self.args.n_agents, combined_belief.shape[-1])
        combined_uncertainty = combined_uncertainty.reshape(evidence.shape[0], self.args.n_agents, 1)
        combined_evidence = self.belief_to_evidence(combined_belief, combined_uncertainty)
        combined_evidence = evidence + 0.1 * combined_evidence
        return combined_evidence

    def belief_to_evidence(self, belief, uncertainty):
        S_a = self.args.n_actions / uncertainty
        # calculate new e_k
        e_a = torch.mul(belief, S_a.expand(belief.shape))
        return e_a

    def combine(self, b0, u0, b1, u1, tt = False):
        # b^0 @ b^(0+1)
        bb = torch.bmm(b0.view(-1, self.args.n_actions, 1), b1.view(-1, 1, self.args.n_actions))
        # b^0 * u^1
        uv1_expand = u1.expand(b0.shape)
        bu = torch.mul(b0, uv1_expand)
        # b^1 * u^0
        uv_expand = u0.expand(b0.shape)
        ub = torch.mul(b1, uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        if tt:
            # calculate b^a
            b_a = (torch.mul(b0, b1) + bu + ub) / ((1 - C).view(-1, 1).expand(b0.shape))
            # calculate u^a
            u_a = torch.mul(u0, u1) / ((1 - C).view(-1, 1).expand(u0.shape))
        else:
            # calculate b^a
            b_a = (torch.mul(b0, b1) + bu + ub) / ((1 - C).view(-1, 1, 1).expand(b0.shape))
            # calculate u^a
            u_a = torch.mul(u0, u1) / ((1 - C).view(-1, 1, 1).expand(u0.shape))
        return b_a, u_a

    def init_hidden(self):
        # trick, create hidden state on same device
        # batch size: 1
        return self.fc1.weight.new_zeros(1, self.args.hidden_dim)
