import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=128):
        super(Actor, self).__init__()
        #Các lớp ẩn
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        #Lớp đầu ra mean và log standard deviation
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.std_log_layer = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        mean = self.mean_layer(a)
        log_std = self.std_log_layer(a)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp() # Fixed standard deviation
        dist = Normal(mean, std)
        u = dist.rsample()  # Reparameterization trick
        action = (torch.tanh(u) + 1) / 2 * self.max_action
        log_prob = dist.log(u) - torch.log(1 - torch.tanh(u)**2)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action.clamp(-self.max_action, self.max_action), log_prob

    def select_action(self, state): #For test
        # Bước 1: Chuẩn bị dữ liệu đầu vào cho PyTorch
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Bước 2: Cho state đi qua mạng để lấy giá trị Mean (bỏ qua log_std)
        mean, _ = self.forward(state)
        
        # Bước 3: Đưa Mean qua hàm Tanh và quy đổi về biên độ thực tế
        action = torch.tanh(mean)
        scaled_action = (action + 1) / 2.0 * self.max_action
        
        # Bước 4: Chuyển đổi ngược từ Tensor về mảng NumPy để môi trường hiểu được
        return scaled_action.detach().cpu().numpy()[0]
