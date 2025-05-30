import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.distributions import kl_divergence


class MAICDelayAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MAICDelayAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = args.n_actions

        self.msg_buffer = [dict() for _ in range(args.n_agents)]

        NN_HIDDEN_SIZE = args.nn_hidden_size
        activation_func = nn.LeakyReLU()

        self.embed_net = nn.Sequential(
            nn.Linear(args.hidden_dim, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2)
        )

        self.inference_net = nn.Sequential(
            nn.Linear(args.hidden_dim + args.n_actions, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2)
        )

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        
        self.msg_net = nn.Sequential(
            nn.Linear(args.hidden_dim + args.latent_dim, NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_actions)
        )

        self.w_query = nn.Linear(args.hidden_dim, args.attention_dim)
        self.w_key = nn.Linear(args.latent_dim, args.attention_dim)
        
    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state, bs, t, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        latent_parameters = self.embed_net(h)
        latent_parameters[:, -self.n_agents * self.latent_dim:] = th.clamp(
            th.exp(latent_parameters[:, -self.n_agents * self.latent_dim:]),
            min=self.args.var_floor)

        latent_embed = latent_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)

        if test_mode:
            latent = latent_embed[:, :self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(latent_embed[:, :self.n_agents * self.latent_dim],
                                    (latent_embed[:, self.n_agents * self.latent_dim:]) ** (1 / 2))
            latent = gaussian_embed.rsample() # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

        h_repeat = h.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        msg = self.msg_net(th.cat([h_repeat, latent], dim=-1)).view(bs, self.n_agents, self.n_agents, self.n_actions)
        
        query = self.w_query(h).unsqueeze(1)
        key = self.w_key(latent).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = th.bmm(query / (self.args.attention_dim ** (1/2)), key).view(bs, self.n_agents, self.n_agents)
        for i in range(self.n_agents):
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        if test_mode:
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0

        gated_msg = alpha * msg

        # insert buffer and get new msg
        # if t < len(self.msg_buffer):
        #     self.msg_buffer[t] = gated_msg
        # else:
        #     self.msg_buffer.append(gated_msg)
        # if t - latency < 0:
        #     d_gated_msg = th.zeros_like(gated_msg)
        # else:
        #     d_gated_msg = self.msg_buffer[i][t]
        if test_mode:
            if self.args.msg_delay_type == 'N_distribution':
                latency = th.normal(mean=self.args.delay_value, std=self.args.delay_scale, size=(1,self.n_agents))
            elif self.args.msg_delay_type == 'constant':
                latency = th.full((1, self.args.n_agents), self.args.delay_value)
            else:
                latency = th.zeros((1, self.args.n_agents))
            latency = th.max(th.tensor(0), latency).int()

            for i in range(self.n_agents):
                self.msg_buffer[i][t + latency[0][i].item()] = gated_msg[:,i,:,:]

            d_gated_msg = th.zeros_like(gated_msg)
            for i in range(self.n_agents):
                if t in self.msg_buffer[i]:
                    d_gated_msg[:,i,:,:] = self.msg_buffer[i].pop(t)
            gated_msg = d_gated_msg


        return_q = q + th.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)

        returns = {}
        if 'train_mode' in kwargs and kwargs['train_mode']:
            if hasattr(self.args, 'mi_loss_weight') and self.args.mi_loss_weight > 0:
                returns['mi_loss'] = self.calculate_action_mi_loss(h, bs, latent_embed, return_q)
            if hasattr(self.args, 'entropy_loss_weight') and self.args.entropy_loss_weight > 0:
                query = self.w_query(h.detach()).unsqueeze(1)
                key = self.w_key(latent.detach()).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
                alpha = F.softmax(th.bmm(query, key), dim=-1).reshape(bs, self.n_agents, self.n_agents)
                returns['entropy_loss'] = self.calculate_entropy_loss(alpha)

        return return_q, h, returns

    def calculate_action_mi_loss(self, h, bs, latent_embed, q):
        latent_embed = latent_embed.view(bs * self.n_agents, 2, self.n_agents, self.latent_dim)
        g1 = D.Normal(latent_embed[:, 0, :, :].reshape(-1, self.latent_dim), latent_embed[:, 1, :, :].reshape(-1, self.latent_dim) ** (1/2))
        hi = h.view(bs, self.n_agents, 1, -1).repeat(1, 1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        
        selected_action = th.max(q, dim=1)[1].unsqueeze(-1)
        one_hot_a = th.zeros(selected_action.shape[0], self.n_actions).to(self.args.device).scatter(1, selected_action, 1)
        one_hot_a = one_hot_a.view(bs, 1, self.n_agents, -1).repeat(1, self.n_agents, 1, 1)
        one_hot_a = one_hot_a.view(bs * self.n_agents * self.n_agents, -1)

        latent_infer = self.inference_net(th.cat([hi, one_hot_a], dim=-1)).view(bs * self.n_agents * self.n_agents, -1)
        latent_infer[:, self.latent_dim:] = th.clamp(th.exp(latent_infer[:, self.latent_dim:]), min=self.args.var_floor)
        g2 = D.Normal(latent_infer[:, :self.latent_dim], latent_infer[:, self.latent_dim:] ** (1/2))
        mi_loss = kl_divergence(g1, g2).sum(-1).mean()
        return mi_loss * self.args.mi_loss_weight

    def calculate_entropy_loss(self, alpha):
        alpha = th.clamp(alpha, min=1e-4)
        entropy_loss = - (alpha * th.log2(alpha)).sum(-1).mean()
        return entropy_loss * self.args.entropy_loss_weight
