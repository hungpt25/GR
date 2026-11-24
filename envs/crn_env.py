import numpy as np

class CogniticeRadioEnv:
    def __init__(self):
        self.N_SU = [1, 5, 10, 15]              #Số lượng SU
        self.P_max = 1                          #Công suất phát tối đa
        self.N = 20                             #Số lượng kênh
        self.A = 10                             #Số kênh bị chiếm bởi PU1
        self.I_pr1 = 0.5                        #Ngưỡng chịu nhiễu
        self.I_pr2 = 0.5
        self.eta = 0.9                          #Hiệu suất thu hoạch năng lượng
        self.T = 1                              #Thời gian trong một khe thời gian
        self.N_0 = 1                            #Nhiễu Gauss
        self.E_max = 0.2                        #Năng lượng tối đa thu thập
        self.C_max = 0.5                        #Năng lượng dự trữ tối đa
        self.epsilon = 1.0                      #Hệ số kênh truyền
        self.w1 = 0.3                           #Phạt năng lượng
        self.w2 = 0.7                           #Phạt nhiễu
        self.beta = 0.5                         #Hệ số cân bằng giữa phần thưởng cá nhân và phần thưởng toàn cục

        self.S = np.zeros((self.N, 1 + self.N_SU[1]))  #Khởi tạo trạng thái ban đầu

        self.data_rate = np.zeros(self.N_SU[1])
        self.personal_reward_combined = np.zeros((self.N, self.N_SU[1]))
        self.personal_reward = np.zeros((self.N, self.N_SU[1]))
        self.global_reward = 0
        self.penalties = np.zeros((self.N, self.N_SU[1]))

        self.P_st = np.zeros((self.N, self.N_SU[1]))  #Công suất phát của SU_tx
        self.P_pt = np.zeros((self.N, 2))  #Công suất phát của PU
        self.k = np.zeros((self.N, self.N_SU[1]))    #Chỉ số hành động SU (0: thu năng lượng + truyền tin, 1: thu năng lượng)
        self.rho = np.zeros((self.N, self.N_SU[1]))   #Hệ số kênh truyền của SU
        self.E_ambient = np.random.uniform(0, self.E_max)  #Năng lượng thu hoạch từ môi trường
        self.energy_used = np.zeros((self.N, self.N_SU[1]))  #Năng lượng sử dụng của SU
        self.noise_to_PU = np.zeros((self.N, self.N_SU[1]))  #Nhiễu gây ra cho PU bởi SU
        self.SU_noise = np.zeros((self.N, self.N_SU[1]))


    def generate_random_channel(self):
        self.g_pt1_st = np.random.exponential(scale=1 / self.epsilon, size=(self.N, self.N_SU[1]))  #Khởi tạo kênh truyền từ PU1 đến SU_tx
        self.g_pt2_st = np.random.exponential(scale=1 / self.epsilon, size=(self.N, self.N_SU[1]))  #Khởi tạo kênh truyền từ PU2 đến SU_tx
        self.g_pt1_sr = np.random.exponential(scale=1 / self.epsilon, size=(self.N, self.N_SU[1]))  #Khởi tạo kênh truyền từ PU1 đến SU_rx
        self.g_pt1_pr1 = np.random.exponential(scale=1 / self.epsilon, size=(self.N))  #Khởi tạo kênh truyền từ PU1 đến PU1
        self.g_pt2_pr2 = np.random.exponential(scale=1 / self.epsilon, size=(self.N))  #Khởi tạo kênh truyền từ PU2 đến PU2
        self.g_st_pr1 = np.random.exponential(scale=1 / self.epsilon, size=(self.N, self.N_SU[1]))  #Khởi tạo kênh truyền từ SU_tx đến PU1
        self.g_st_pr2 = np.random.exponential(scale=1 / self.epsilon, size=(self.N, self.N_SU[1]))  #Khởi tạo kênh truyền từ SU_tx đến PU2
        self.g_st_sr = np.random.exponential(scale=1 / self.epsilon, size=(self.N, self.N_SU[1], self.N_SU[1]))  #Khởi tạo kênh truyền từ SU_tx đến SU_rx


    def get_local_observation(self,time_step, su_index):
        self.S[0][su_index] = np.concatenate((self.E[time_step - 1][su_index] if time_step > 0 else 0, 
                                           self.C[time_step][su_index], 
                                           self.g_pt1_st[time_step][su_index], 
                                           self.g_pt2_st[time_step][su_index],
                                           self.g_st_sr[time_step][su_index][su_index], 
                                           self.g_pt1_sr[time_step][su_index], 
                                           self.g_st_pr1[time_step][su_index], 
                                           self.g_st_pr2[time_step][su_index]))
        return self.S[0][su_index]


    def reset(self):
        self.E = np.zeros((self.N, self.N_SU[1]))  #Khởi tạo năng lượng thu hoạch ban đầu
        self.C = np.zeros((self.N, self.N_SU[1]))  #Khởi tạo năng lượng dự trữ ban đầu
        self.generate_random_channel()
        self.v = [1] * self.A + [0] * (self.N - self.A)  #Kênh đầu tiên bị chiếm bởi PU1, các kênh còn lại chiếm bởi PU2
        self.S[0] = [self.v[0], *[self.get_local_observation(0, su_index) for su_index in range(self.N_SU[1])]]
        return self.S[0]


    def calculate_energy_harvested(self, time_step, SU_index):
        return (self.rho[time_step][SU_index] * 
                self.T * 
                sum(self.P_pt[time_step]) * 
                self.eta * 
                (self.g_pt1_st[time_step][SU_index] + self.g_pt2_st[time_step][SU_index]) + 
                self.E_ambient[time_step])


    def calculate_penalties(self, time_step, SU_index):
        return (self.w1 * max(0, self.energy_used[time_step][SU_index] - self.C[time_step][SU_index]) + 
                self.w2 * max(0, self.noise_to_PU[time_step][SU_index] - (self.I_pr1 if time_step < 11 else self.I_pr2)))  #Tính toán phạt năng lượng và phạt nhiễu


    def step(self, actions, time_step, PU_power): #action: [k_i_t, P_sti_t, p_i_t]
        for i in range(self.N_SU[1]):
            self.P_st[time_step][i] = actions[i][1]  #Công suất phát của SU_tx
            self.k[time_step][i] = actions[i][0]   #Chỉ số hành động SU
            self.rho[time_step][i] = actions[i][2]  #Hệ số kênh truyền của SU
            self.P_pt = PU_power  #Công suất phát của PU
            

            #Năng lượng thu hoạch của SU
            self.E[time_step][i] = self.calculate_energy_harvested(time_step, i)
            #Năng lượng sử dụng của SU
            self.energy_used[time_step][i] = (1 - self.k[time_step][i]) * self.P_st[time_step][i] * self.T  * (1 - self.rho[time_step][i]) 

            #Nang lượng dự trữ của SU ở đầu khe thời gian tiếp theo (t+1) = min(Năng lượng dự trữ tối đa, Năng lượng dự trữ ở đầu khe thời gian hiện tại + Năng lượng thu hoạch - Năng lượng sử dụng)
            self.C[time_step + 1][i] = min(self.C_max, self.C[time_step][i] + self.E[time_step][i] - self.energy_used[time_step][i])
           
            #Noise của các SU khác  
            for j in range(0, i):
                self.SU_noise[time_step][i] += (1 - self.rho[time_step][j]) * self.P_st[time_step][j] * self.g_st_sr[time_step][j][i]
            for j in range(i + 1, self.N_SU[1]):
                self.SU_noise[time_step][i] += (1 - self.rho[time_step][j]) * self.P_st[time_step][j] * self.g_st_sr[time_step][j][i]
            
            #Signal to Noise plus Interference Ratio (SINR) của SU
            SINR_1 = ((1 - self.rho[time_step][i]) * self.P_st[time_step][i] * self.g_st_sr[time_step][i][i]) / (self.N_0 + self.P_pt[time_step][0] * self.g_pt1_sr[time_step][i] + self.SU_noise[time_step][i])
            SINR_2 = ((1 - self.rho[time_step][i]) * self.P_st[time_step][i] * self.g_st_sr[time_step][i][i]) / (self.N_0 + self.SU_noise[time_step][i])
            
            #Data rate của SU
            if time_step <11: #Tính toán tốc độ dữ liệu của SU khi PU1 đang hoạt động
                self.data_rate[time_step][i] = self.T * np.log2(1 + SINR_1)  
                #Tính toán nhiễu gây ra cho PU1 bởi SU
                self.noise_to_PU[time_step][i] = (1 - self.rho[time_step][i]) * self.P_st[time_step][i] * self.g_st_pr1[time_step][i]  
            else: #Tính toán tốc độ dữ liệu của SU khi PU2 đang hoạt động
                self.data_rate[time_step][i] = self.T * np.log2(1 + SINR_2)  
                #Tính toán nhiễu gây ra cho PU2 bởi SU
                self.noise_to_PU[time_step][i] = (1 - self.rho[time_step][i]) * self.P_st[time_step][i] * self.g_st_pr2[time_step][i]  

            #Thưởng cá nhân của SU
            if self.k[time_step][i] == 0 and (1 - self.rho[time_step][i]) * self.P_st[time_step][i] * self.T <= self.C[time_step][i] and self.noise_to_PU[time_step][i] <= self.I_pr1:
                self.personal_reward[time_step][i] = self.data_rate[time_step][i]  #Nếu SU ko gây nhiễu cho PU, thưởng của SU sẽ bằng data rate
            elif self.k[time_step][i] == 1 and (1 - self.rho[time_step][i]) * self.P_st[time_step][i] * self.T > self.C[time_step][i]:
                self.personal_reward[time_step][i] = 0  #Nếu SU thu thập năng lượng, thưởng của SU sẽ bằng 0
            else: self.personal_reward[time_step][i] = -1 #Thưởng = -1 với các trường hợp còn lại

            self.global_reward += self.data_rate[time_step][i] #Thưởng toàn cục
            self.penalties[time_step][i] = self.calculate_penalties(time_step, i)


        #Tính toán phần thưởng cá nhân của SU
        for i in range(self.N_SU[1]):
            self.personal_reward_combined[time_step][i] = self.beta * self.global_reward + (1 - self.beta) * (self.personal_reward[time_step][i] - self.penalties[time_step][i])  

        #Cập nhật trạng thái của SU ở đầu khe thời gian tiếp theo (t+1)
        self.S[time_step + 1] = [self.v[time_step + 1], *[self.get_local_observation(time_step + 1, su_index) for su_index in range(self.N_SU[1])]]

        if time_step == self.N - 1:
            done = True

        return self.S[time_step + 1], self.personal_reward_combined[time_step], self.global_reward, done
