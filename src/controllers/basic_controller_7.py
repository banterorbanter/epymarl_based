from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNN_msg(nn.Module):
    def __init__(self, input_shape, hidden_shape, action_shape):
        super(RNN_msg, self).__init__()
        self.hidden_shape = hidden_shape
        self.fc1 = nn.Linear(input_shape, hidden_shape)
        self.rnn = nn.GRUCell(hidden_shape, action_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_shape).zero_()

    def forward(self, hidden_state, input_h):
        x = F.elu(self.fc1(input_h))
        h_in = hidden_state.reshape(-1, self.hidden_shape)
        h = self.rnn(x, h_in)
        return h, h


# This multi-agent controller shares parameters between agents
class BasicMAC_7:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_action = args.n_actions
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.match_weight = 0.0
        self.msg_rnn = RNN_msg(64,self.n_action,self.n_action)
        self.epsilon = 1e-10
        self.hidden_states = None
        self.transmit_gap = th.zeros((args.batch_size_run * self.n_agents))
        self.receive_gap = th.zeros((args.batch_size_run,self.n_agents,self.n_agents))
        self.msg_old_test = th.zeros((args.batch_size_run * self.n_agents,self.n_action))
        self.msg_old_test_reshape = th.zeros((args.batch_size_run,self.n_agents,self.n_agents,self.n_action))
        
        self.dummy_buffer = [dict() for _ in range(args.n_agents)]

        ##################hyperparameters######################

        self.fresh_limit = self.args.fresh_limit*2    # validation period for the messages in the received message buffer
        self.transmit_limit = self.args.transmit_limit   # if the transmitter has not sent within this limit, the transmitter will send
        self.delta = self.args.delta    # delta in Algorithm 1
        #######################################################
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_local_outputs, input_hidden_states, vi = self.forward(ep_batch, t_ep, test_mode=test_mode)        
        # store all the actions
        input_hidden_states = input_hidden_states.view(-1,64)
        self.hidden_states_msg, dummy = self.msg_rnn(self.hidden_states_msg, input_hidden_states)
        dummy = dummy.reshape(ep_batch.batch_size,self.n_agents,self.n_action)

        if self.args.msg_delay_type == 'N_distribution' and test_mode:
            latency = th.normal(mean=self.args.delay_value, std=self.args.delay_scale, size=(1, self.n_agents))
        elif self.args.msg_delay_type == 'constant' and test_mode:
            latency = th.full((1, self.args.n_agents), self.args.delay_value)
        else:
            latency = th.zeros((1, self.args.n_agents))
        latency = th.max(th.tensor(0), latency).int()

        for i in range(self.n_agents):
            self.dummy_buffer[i][t_ep + latency[0][i].item()] = dummy[:, i, :]

        d_dummy = th.zeros_like(dummy).cuda()
        for i in range(self.n_agents):
            if t_ep in self.dummy_buffer[i]:
                d_dummy[:, i, :] = self.dummy_buffer[i].pop(t_ep)

        d_dummys = th.zeros((ep_batch.batch_size, self.n_action)).cuda()
        for i in range(self.n_agents):
            d_dummys += d_dummy[:, i, :].cuda()

        agent_global_outputs = th.zeros((ep_batch.batch_size, self.n_agents, self.n_action)).cuda()
        for i in range(self.n_agents):
            agent_global_outputs[:, i, :] = ((d_dummys - d_dummy[:, i, :]) / float(self.n_agents - 1)).view(ep_batch.batch_size,
                                                                                                            self.n_action).cuda()

        agent_outputs = agent_local_outputs + agent_global_outputs.cuda()
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return chosen_actions

    def select_actions_noisy_env(self, loss_pattern, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_local_outputs, input_hidden_states, vi = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # store all the actions
        input_hidden_states = input_hidden_states.view(-1,64)
        self.hidden_states_msg, dummy = self.msg_rnn(self.hidden_states_msg, input_hidden_states)

        ###################send message to teammates#############################
        ind = (((dummy-self.msg_old_test)**2).sum(1) > self.delta).float()   # dd is (48,1)
        self.transmit_gap[ind==0] = self.transmit_gap[ind==0] + 1
        ind[self.transmit_gap>=self.transmit_limit] = 1
        self.transmit_gap[ind==1] = 0
        self.msg_old_test[ind==1,:] = dummy[ind==1,:]   # for the messages where dd = 1, transmit it
        #########add loss to the transmitted message################
        ind = ind.reshape(8,7)
        ind = th.stack([ind]*7,2)   #(8,6,6)
        loss_pattern = loss_pattern.reshape((8,7,7))
        ind = ind * loss_pattern
        ########check invalid messages in the receiver buffer#######
        self.receive_gap[ind==1] = 0
        self.receive_gap[ind==0] = self.receive_gap[ind==0] + 1
        use_old = 1-ind
        use_old[self.receive_gap>self.fresh_limit] = 0
        ########assign the packets to all the receiving agents######
        ind = th.stack([ind]*8,3).float()   # (8,6,6,14)
        use_old = th.stack([use_old]*8,3).float()   # (8,6,6,14)
        dummy = th.stack([dummy]*6,1).float()   # (48,6,14)
        dummy = dummy.reshape(8,7,7,8)  
        dummy_final = self.msg_old_test_reshape * use_old + dummy * ind  
        self.msg_old_test_reshape = dummy_final
        #########
        agent0 = (dummy_final[:,1,0,:] + dummy_final[:,2,0,:] + dummy_final[:,3,0,:] + dummy_final[:,4,0,:] + dummy_final[:,5,0,:])/5.0
        agent1 = (dummy_final[:,0,1,:] + dummy_final[:,2,1,:] + dummy_final[:,3,1,:] + dummy_final[:,4,1,:] + dummy_final[:,5,1,:])/5.0
        agent2 = (dummy_final[:,0,2,:] + dummy_final[:,1,2,:] + dummy_final[:,3,2,:] + dummy_final[:,4,2,:] + dummy_final[:,5,2,:])/5.0
        agent3 = (dummy_final[:,0,3,:] + dummy_final[:,1,3,:] + dummy_final[:,2,3,:] + dummy_final[:,4,3,:] + dummy_final[:,5,3,:])/5.0
        agent4 = (dummy_final[:,0,4,:] + dummy_final[:,1,4,:] + dummy_final[:,2,4,:] + dummy_final[:,3,4,:] + dummy_final[:,5,4,:])/5.0
        agent5 = (dummy_final[:,0,5,:] + dummy_final[:,1,5,:] + dummy_final[:,2,5,:] + dummy_final[:,3,5,:] + dummy_final[:,4,5,:])/5.0
        
        agent_global_outputs =th.cat((agent0.view((8,1,14)),agent1.view((8,1,14)),agent2.view((8,1,14)),agent3.view((8,1,14)),agent4.view((8,1,14)),agent5.view((8,1,14))),1)
        
        agent_outputs = agent_local_outputs + agent_global_outputs
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        return chosen_actions
    
    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs, visibility_matrix = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)    # this is (48,98)!!!!!!!!!
        
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs_t = agent_outs.clone()
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs_t
                            + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.clone().view(ep_batch.batch_size, self.n_agents, -1), self.hidden_states.view(ep_batch.batch_size, self.n_agents, -1), visibility_matrix

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.hidden_states_msg = self.msg_rnn.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def set_match_weight(self, weight):
        self.match_weight = weight

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.msg_rnn.load_state_dict(other_mac.msg_rnn.state_dict())
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.msg_rnn.cuda()
        
    def save_models(self, path):
        th.save(self.msg_rnn.state_dict(), "{}/msg_rnn.th".format(path))
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.msg_rnn.load_state_dict(th.load("{}/msg_rnn.th".format(path), map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

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
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)   
        visibility_matrix = th.ones((1))
        
        return inputs, visibility_matrix

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape



