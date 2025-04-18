import math
import time
import numpy as np
from collections import deque
from humanoid.envs import neoCfg
import torch
import matplotlib.pyplot as plt

class config:
    cycle_time = 1.5
# 创建一个输入，其中 sin 和 cos 不为 0，其他值全为 0
def generate_custom_input(count_lowlevel, cfg, policy_output, joint_pos_scaled):
    obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
    obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim.dt / config.cycle_time)  # sin 不为 0
    obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim.dt / config.cycle_time)  # cos 不为 0
    obs[0,5:15] = joint_pos_scaled
    # obs[0, 7] = joint_pos_scaled[2]#0#max(0,math.sin(2 * math.pi * count_lowlevel * cfg.sim.dt / config.cycle_time))*0.25#
    # obs[0, 8] = joint_pos_scaled[0,3]#0#max(0,math.cos(2 * math.pi * count_lowlevel * cfg.sim.dt / config.cycle_time))*0.5
    # obs[0, 9] = joint_pos_scaled[0,4]#0#max(0,math.sin(2 * math.pi * count_lowlevel * cfg.sim.dt / config.cycle_time))*0.25
    # obs[0, 12] = joint_pos_scaled[0,7]#0#min(0,math.sin(2 * math.pi * count_lowlevel * cfg.sim.dt / config.cycle_time))*0.25
    # obs[0, 13] = joint_pos_scaled[0,8]#min(0,math.cos(2 * math.pi * count_lowlevel * cfg.sim.dt / config.cycle_time))*0.5
    # obs[0, 14] = joint_pos_scaled[0,9]#min(0,math.sin(2 * math.pi * count_lowlevel * cfg.sim.dt / config.cycle_time))*0.2
    # obs[0, 25:35] = policy_output
    return obs

# 打包 15 次的输入
def generate_stacked_input(count_lowlevel, cfg,policy_output, joint_pos_scaled):
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):  # 假设 frame_stack 为 15
        custom_input = generate_custom_input(count_lowlevel, cfg,policy_output, joint_pos_scaled)
        hist_obs.append(custom_input)
    # 将打包的输入转换为模型的输入格式
    policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
    for i in range(cfg.env.frame_stack):
        policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
    return policy_input

# 示例：生成输入并传递给模型
def test_model_with_custom_input(policy, cfg, count_lowlevel, joint_pos_scaled, policy_output):
    policy_input = generate_stacked_input(count_lowlevel, cfg, policy_output, joint_pos_scaled)
    policy_output = policy(torch.tensor(policy_input)).detach().cpu().numpy().squeeze()
    joint_pos_scaled = policy_output * 0.25
    return policy_output, joint_pos_scaled

# 绘制策略输出
def plot_policy_output(joint_pos):
    joint_pos = np.array(joint_pos)  # 转换为 NumPy 数组
    time_steps = np.arange(len(joint_pos))*cfg.sim.dt  # 时间步
    plt.figure(figsize=(10, 6))
    for i in range(joint_pos.shape[1]):  # 遍历每个关节
        plt.plot(time_steps, joint_pos[:, i], label=f"Joint {i+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Joint Positions")
    plt.title("Policy Output Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("policy_output_over_time.png")  # 保存图像
    plt.show()

if __name__ == '__main__':
    # 示例：测试模型
    policy = torch.jit.load("/home/wx/humanoid/humanoid-gym/logs/neo_ppo/exported/policies_3-29/policy_1.pt")
    cfg =neoCfg()
    count_lowlevel = 0  # 假设从第 0 步开始
    joint_pos = []
    joint_pos_scaled=np.zeros([1,10])
    policy_output=np.zeros([1,10])
    policy_output_list = []
    try:
        while True:##count_lowlevel < 1000:
            start_time = time.monotonic()
            policy_output, joint_pos_scaled = test_model_with_custom_input(
                policy, cfg, count_lowlevel, joint_pos_scaled, policy_output
            )
            policy_output_list.append(policy_output)
            joint_pos.append(joint_pos_scaled)
            print(joint_pos_scaled)
            end_time = time.monotonic()
            if end_time - start_time < cfg.sim.dt:
                time.sleep(cfg.sim.dt - (end_time - start_time))
                #print("delay")
            count_lowlevel += 1  # 假设每一步都需要更新输入
    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        # 绘制策略输出
        plot_policy_output(policy_output_list)
        plot_policy_output(joint_pos)
        