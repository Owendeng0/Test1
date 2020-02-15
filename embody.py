import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


states = ["急加速", "缓加速", "稳态加速", "缓加速", "急减速"]  #五个隐状态
n_states = len(states)

observations = ["L1", "L2", "L3", "L4", "L5"]  #五个观测序列
n_observations = len(observations)

start_probability = np.ones(5)/5   #初始概率

#状态转移矩阵
transition_probability = np.array([
    [0.5, 0.25, 0.15, 0.05, 0.05],
    [0.2, 0.5, 0.2, 0.05, 0.05],
    [0.05, 0.2, 0.5, 0.2, 0.05],
    [0.05, 0.05, 0.2, 0.5, 0.2],
    [0.05, 0.05, 0.15, 0.25, 0.5]
])

#观测状态概率矩阵
emission_probability = np.array([
    [0.05, 0.05, 0.1, 0.25, 0.55],
    [0.05, 0.05, 0.15, 0.55, 0.2],
    [0.1, 0.2, 0.5, 0.2, 0.1],
    [0.2, 0.55, 0.15, 0.05, 0.05],
    [0.6, 0.2, 0.1, 0.05, 0.05]
])


model = hmm.MultinomialHMM(n_components = n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionpro_ = emission_probability


"""
print(transition_probability)
print("-"*20)
print(emission_probability)
"""
#计算后向概率
def backward_probalility(first_states,states, transition_probability, emission_probability):
    #获取下一时刻的隐状态概率
    for i in range(len(states)):
        if first_states == states[i]:
            tr_prob = transition_probability[i]
            break
    
    #获取各个位置的概率
    emission_probability = np.asmatrix(emission_probability)
    tr_prob = np.asmatrix(tr_prob)
    position_prob = tr_prob * emission_probability
    position_prob = np.asarray(position_prob)
    position_prob = position_prob.flatten()
    return position_prob

position_prob_B = backward_probalility("稳态加速", states, transition_probability, emission_probability)
position_prob_A = backward_probalility("急加速", states, transition_probability, emission_probability)

def print_positon(observations, position_prob):
    for ob, prob in zip(observations, position_prob):
        print("处于位置{}的概率为：{}".format(ob, prob))

print("对于A车而言:"+"-"*40)
print_positon(observations, position_prob_A)
print("\n")
print("对于B车而言：" + "-"*40)
print_positon(observations, position_prob_B)
print("-"*50)
#计算冲突概率
class Crash:
    def __init__(self, set_warning_prob, position_prob_A, position_prob_B):
        self.set_warning_prob = set_warning_prob
        self.position_prob_A = position_prob_A
        self.position_prob_B = position_prob_B
        self.crash_prob = np.sum(self.position_prob_A[:3] * self.position_prob_B[2:] )
    
    #判断是否要预警
    def judge_crash(self):
        if self.crash_prob >= self.set_warning_prob:
            print("waring!!!")
        else:
            print("It's save.")     
    
    #各个碰撞点的概率        
    def get_position_crash_prob(self):
        position_crash_prob = self.position_prob_A[:3] * self.position_prob_B[2:]
        return position_crash_prob
    
    #绘制出各个位置的碰撞概率热力图
    def draw_prob(self):
        postition_crash_prob = self.get_position_crash_prob()
        position_crash_prob = np.round(crash.get_position_crash_prob(), 4)
        crash_np_data = np.vstack((np.hstack(([0,0], position_crash_prob)), np.zeros(5)))
        crash_data = pd.DataFrame(crash_np_data, columns = ["L1", "L2", "L3", "L4", "L5"], index = ["A", "B"])
        fig, ax = plt.subplots(figsize = (9,4)) #设置画面大小
        sns.heatmap(crash_data, annot = True,vmax = 0.03, vmin = 0.02, square = True , cmap = "Reds")
        plt.savefig("碰撞概率图.png")
        plt.show()

crash = Crash(0.05,position_prob_A, position_prob_B )
print("碰撞的概率为：{:.4f}".format(crash.crash_prob))
crash.judge_crash()
crash.draw_prob()
