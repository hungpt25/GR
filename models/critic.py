import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim, total_action_dim, hidden_dim=128):
        super(CentralizedCritic, self).__init__()
        
        # --- Q-Network 1 ---
        self.l1_q1 = nn.Linear(global_state_dim + total_action_dim, hidden_dim)
        self.l2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # --- Q-Network 2 (Twin Q-Network để chống overestimation) ---
        self.l1_q2 = nn.Linear(global_state_dim + total_action_dim, hidden_dim)
        self.l2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, global_state, joint_actions):
        x = torch.cat([global_state, joint_actions], dim=-1)
        
        # Lan truyền qua Q1
        q1 = F.relu(self.l1_q1(x))
        q1 = F.relu(self.l2_q1(q1))
        q1 = self.q1_out(q1)
        
        # Lan truyền qua Q2
        q2 = F.relu(self.l1_q2(x))
        q2 = F.relu(self.l2_q2(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2