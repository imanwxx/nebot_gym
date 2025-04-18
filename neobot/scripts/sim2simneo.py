# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import neoCfg
import torch
import time
import matplotlib.pyplot as plt 
import os

class cmd:
    vx = 0.
    vy = 0.
    dyaw = 0.
class joint:
    joint_names={
        '0': 'l_hip_roll',
        '1': 'l_hip_yaw',
        '2': 'l_hip_pitch',
        '3': 'l_knee',
        '4': 'l_ankle_pitch',
        '5': 'r_hip_roll',
        '6': 'r_hip_yaw',
        '7': 'r_hip_pitch',
        '8': 'r_knee',
        '9': 'r_ankle_pitch',}
def plt_fuctioon(list1,list2,names,cfg):
    time_steps = np.arange(list1.shape[0])*cfg.sim_config.dt
    
    for i in range(list1.shape[1]):
        sine_wave = 0.25 * np.sin(2 * np.pi * time_steps / cfg.cycle_time)
        plt.figure(figsize=(10, 8))
        
        if i==0 or i==1 or i==5 or i==6:
            plt.axhline(y=0, color='r', linestyle='-', label='reference_line')
        elif i==2:
            sine_wave = 0.25 * np.sin(2 * np.pi * (time_steps / cfg.cycle_time-1/2))
            sine_wave =np.maximum(0, sine_wave)
            plt.plot(time_steps, sine_wave, color='r', linestyle='-', label='reference_line')
        elif i==3:
            sine_wave = 0.25 * np.sin(2 * np.pi * (time_steps / cfg.cycle_time-1/2))
            sine_wave = -sine_wave*2
            sine_wave = np.minimum(0, sine_wave)
            plt.plot(time_steps, sine_wave, color='r', linestyle='-', label='reference_line')
        elif i==4:
            sine_wave = -0.25 * np.sin(2 * np.pi * (time_steps / cfg.cycle_time-1/2))
            sine_wave = np.minimum(0, sine_wave)
            plt.plot(time_steps, sine_wave, color='r', linestyle='-', label='reference_line')
        elif i==7:
            sine_wave =np.maximum(0, sine_wave)
            plt.plot(time_steps, sine_wave, color='r', linestyle='-', label='reference_line')
        elif i==8:
            sine_wave = -0.5 * np.sin(2 * np.pi * (time_steps / cfg.cycle_time))
            sine_wave=np.minimum(0, sine_wave)
            plt.plot(time_steps, sine_wave, color='r', linestyle='-', label='reference_line')
        elif i==9:
            sine_wave =np.maximum(0, sine_wave)
            plt.plot(time_steps, sine_wave, color='r', linestyle='-', label='reference_line')
        else:
            sine_wave =np.minimum(0, sine_wave)
            plt.plot(time_steps, -2*sine_wave, color='r', linestyle='-', label='reference_line')
        plt.plot(time_steps, list1[:,i], label="target_"+names[f"{i}"])
        plt.plot(time_steps, list2[:,i], label="output_"+names[f"{i}"])
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel(names[f"{i}"])
        plt.title(names[f"{i}"]+' over time')
        plt.show()
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):#15
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double)) #45

    count_lowlevel = 0
    joint_angles = []
    
    
    try:
        #for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        for _ in tqdm(range(int(5000)), desc="Simulating..."):
            control_start_time = time.time()
            # Obtain an observation
            q, dq, quat, v, omega, gvec = get_obs(data)#关节角度*10、角速度*10、imu角度*3、线速度、imu角速度*3、重力向量
            q = q[-cfg.env.num_actions:]
            dq = dq[-cfg.env.num_actions:]
            
            joint_angles.append(q.copy())
            
            # print("关节角度: ", q)
            # 1000hz -> 100hz,低频推理，高频控制，每decimation次进行一次推理
            if count_lowlevel % cfg.sim_config.decimation == 0:
                
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi#变到-pi到pi之间

                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * Sim2simCfg.sim_config.dt / cfg.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * Sim2simCfg.sim_config.dt / cfg.cycle_time)
                obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 5:15] = q * cfg.normalization.obs_scales.dof_pos
                
                obs[0, 15:25] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 25:35] = action
                obs[0, 35:38] = omega
                obs[0, 38:41] = eu_ang
                # print("obs: ", obs)
                #print("关节角度: ", q)
                #print("角速度: ", dq)
                #print("imu角速度: ", omega)
                #print("imu角度: ", eu_ang)
                #time.sleep(1)
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()
                # print("hist_obs_shape: ", hist_obs)
                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
                #print("policy_input_shape: ", policy_input.shape)
                # print("policy_input: ", policy_input)
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                # print("action: ", action)
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                
                target_q = action * 0.25#cfg.control.action_scale
                
            cfg.action_list.append(target_q.copy())
            #target_q等上一次的目标角度执行PD控制，每decimation次更新一次目标角度
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                            target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau
            print("mujoco关节力矩: ", tau)
            
            # if (control_end_time - control_start_time) < cfg.sim_config.dt:
            #     time.sleep(cfg.sim_config.dt - (control_end_time - control_start_time))
            # # print("sim time: ", time.time() - control_end_time)
            # print("control_frequency: ", 1 / (time.time() - control_start_time))

            mujoco.mj_step(model, data)
            viewer.render()
            control_end_time = time.time()
            if control_end_time - control_start_time < 0.001:
                time.sleep(0.001 - (control_end_time - control_start_time))
                print("control_frequency: ", 1 / (time.time() - control_start_time))
            count_lowlevel += 1
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        viewer.close()
        
        # 模型输出动作曲线
        action_list = np.array(cfg.action_list)
        time_steps = np.arange(action_list.shape[0]) * cfg.sim_config.dt*cfg.sim_config.decimation  # 时间步长
        plt.figure(figsize=(10, 6))
        for i in range(action_list.shape[1]):
            # if i==5 or i==6:
            #     plt.plot(time_steps, action_list[:, i], label="Joint hip roll" if i == 5 else "Joint hip yaw")
            if i==7 or i==9:
                plt.plot(time_steps, action_list[:, i], label="Joint hip pitch" if i == 7 else "Joint ankle")
            if i==8:
                plt.plot(time_steps, action_list[:, i], label="Joint knee")
        plt.xlabel("Time (s)")
        plt.ylabel("Action")
        plt.title("Action Over Time(mujoco)")
        plt.grid()
        plt.legend()
        plt.show()
        print("动作曲线已保存为 Action Over Time(mujoco).png")
        # 绘制关节角度图像
        np.savetxt("joint_angles.csv", np.array(joint_angles), delimiter=",")
        print("关节角度数据已保存到 joint_angles.csv")
        joint_angles = np.array(joint_angles)
        plt_fuctioon(action_list,joint_angles,joint.joint_names,cfg)
        time_steps = np.arange(joint_angles.shape[0]) * cfg.sim_config.dt  # 时间步长
        # 绘制右腿髋关节回转&偏航关节角度图像
        plt.figure(figsize=(10, 6))
        for i in range(joint_angles.shape[1]):
            if i==5 or i==6:
                plt.plot(time_steps, joint_angles[:, i], label="Joint hip roll" if i == 5 else "Joint hip yaw")

        plt.xlabel("Time (s)")
        plt.ylabel("Right_Leg Joint Hip roll & hip yaw Angles (rad)")
        plt.title("Right_Leg Joint Hip roll & hip yaw Angles Over Time(mujoco)")
        plt.legend()
        plt.grid()
        parts = args.load_model.split("/")
        for part in parts:
            if "policies" in part:
                dir_name =  part
                os.makedirs(f"./imgs/mujoco_{dir_name}", exist_ok=True)
                plt.savefig(f"./imgs/mujoco_{dir_name}/Right_Leg Hip roll & hip yaw Angles Over Time(mujoco).png")
        plt.show()
        print("关节角度图像已保存为Right_Leg Hip roll & hip yaw Angles Over Time(mujoco).png")
        # 生成正弦函数数据
        sine_wave = 0.25 * np.sin(2 * np.pi * time_steps / cfg.cycle_time)
        sine_wave = np.maximum(sine_wave, 0)  # 确保正弦函数为非负值
        plt.figure(figsize=(10, 6))
        for i in range(joint_angles.shape[1]):
            if i == 7 or i == 9:
                plt.plot(time_steps, joint_angles[:, i], label="Joint hip pitch" if i == 7 else "Joint ankle")
        
        # 添加正弦函数曲线
        plt.plot(time_steps, sine_wave, label="Sine Wave (0.64s period, 0.25 amplitude)", linestyle="--", color="black")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint Angles (rad)")
        plt.title("Joint Angles and Sine Wave Over Time(mujoco)")
        plt.legend()
        plt.grid()
        plt.savefig(f"./imgs/mujoco_{dir_name}/Right_Leg Hip pitch & ankle Joint Angles and Sine Wave Over Time(mujoco).png")
        plt.show()
        print("关节角度与正弦函数图像已保存为 joint_angles_with_sine_wave(mujoco).png")

        #绘制膝关节角度与正弦参考函数对比图
        sine_wave = 0.5 * np.sin(2 * np.pi * time_steps / cfg.cycle_time)  # 生成正弦参考函数
        sine_wave = -np.maximum(sine_wave, 0)
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, joint_angles[:, 8], label=f"Joint knee")
        plt.plot(time_steps, sine_wave, label="Sine Wave (0.64s period, 0.25 amplitude)", linestyle="--", color="black")
        plt.xlabel("Time (s)")
        plt.ylabel("Right_Leg Knee Joint Angles (rad)")
        plt.title("Right_Leg Knee Joint Angles and Sine Wave Over Time(mujoco)")
        plt.legend()
        plt.grid()
        plt.savefig(f"./imgs/mujoco_{dir_name}/Right_Leg Knee Joint Angles and Sine Wave Over Time(mujoco).png")
        plt.show()
        print("关节角度与正弦函数图像已保存为 Right_Leg kneeJoint Angles and Sine Wave Over Time(gym).png")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()
    
    class Sim2simCfg(neoCfg):
        cycle_time = 0.64
        action_list = []
        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/neobot_no_mass_matrix/mjcf/scene.xml'
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L.xml'
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/loong/mjcf/scene.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 10#10

        # class robot_config:
        #     kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
        #     kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
        #     tau_limit = 200. * np.ones(12, dtype=np.double)
        class robot_config:
            kps = np.array([50, 50, 100,100, 10, 50, 50, 100,100, 10], dtype=np.double)#*0.5
            kds = np.array([2, 2, 4, 4, 0.5, 2, 2, 4, 4, 0.5], dtype=np.double)#*0.5
            tau_limit = 7*np.ones(10, dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
