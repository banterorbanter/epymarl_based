# Modified from attend rnn agent. Can be merged with a parameter to select.
from types import SimpleNamespace

import torch.nn
import torch.nn as nn

from .agent import Agent


class EntityCattingRNNAgent(Agent):
    def __init__(self, input_scheme, args: SimpleNamespace):
        super().__init__(input_scheme, args)

        self.args = args
        self.device = args.device
        self.n_heads: int = getattr(args, "agent_attn_heads")   # Attention heads.
        self.hidden_dim: int = getattr(args, "agent_hidden_dim")    # Dimension of embedding andRNN.
        self.attn_dim: int = getattr(args, "agent_attn_dim")    # Dimension of full attention layer
        self.gru_layers: int = getattr(args, "agent_gru_layers")    # Number of GRU layers.

        # Embedding layers: scheme -> hidden_dim
        self.n_entities = 0
        self.embedding_layers = nn.ModuleList()
        for feat_name, feat_shape in input_scheme[0].items():
            self.embedding_layers.append(
                nn.Linear(feat_shape[1], self.hidden_dim, bias=False, device=self.device)
            )
            self.n_entities += feat_shape[0]


        # Embedding layers: 1 -> hidden_dim
        for feat_name, feat_shape in input_scheme[1].items():
            self.embedding_layers.append(
                nn.Embedding(feat_shape[1], self.hidden_dim, device=self.device)
            )
            self.n_entities += feat_shape[0]

        # Encoding layers: hidden_dim -> attn_dim
        self.encoding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attn_dim, device=self.device),
            nn.LeakyReLU(inplace=True),
        )

        # Output layers: attn_dim -> hidden_dim -> n_actions q
        self.rnn_proj = nn.Linear(self.attn_dim * self.n_entities, self.hidden_dim, device=self.device)
        # TODO: RNN might not be advanced. Transformer decoder seems to work here.
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=self.gru_layers, batch_first=True, device=self.device)
        self.decoding = nn.Linear(self.hidden_dim, args.n_actions, device=self.device)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(self.gru_layers, self.args.agent_hidden_dim, device=self.device)
        # return self.own_embed.weight.new(1, self.args.agent_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # TODO: For simple implementation, all features are considered as entities.
        #       Actually, move feats and additional embeddings should merge into own features.
        #       According to experiments, it seems that the influence exists but not much.

        # Embedding: scheme -> hidden_dim
        entities = torch.cat(
            [layer(feature_input) for feature_input, layer in zip(inputs, self.embedding_layers)],
            dim=-2
        )
        # Encoding: hidden_dim -> attn_dim
        entities = self.encoding(entities)

        # Feature catting.
        attn = entities.reshape(*entities.shape[:-2], -1)

        batch_size, time_size, n_agents, _ = attn.shape

        # Output:  attn_dim -> n_actions
        x = self.rnn_proj(attn)     # attn_dim -> hidden_dim

        x = x.transpose(1, 2).reshape(batch_size * n_agents, time_size, self.hidden_dim)  # b * t * n * d -> b * n * t * d -> (b * n) * t * d
        h = hidden_state.reshape(self.gru_layers, batch_size * n_agents, self.hidden_dim)

        x, h = self.rnn(x, h)   # GRU forward.

        x = x.reshape(batch_size, n_agents, time_size, self.hidden_dim).transpose(1, 2)
        h = h.reshape(self.gru_layers, batch_size, n_agents, self.hidden_dim)

        q = self.decoding(x)

        return q, h
