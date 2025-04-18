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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class neoCfg(LeggedRobotCfg):
    """
    Configuration class for the Alex humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 41
        num_observations = int(frame_stack * num_single_obs)#在base_task中令num_obs=num_observations
        single_num_privileged_obs = 73-8
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 10#自由度
        num_envs = 4096#环境数
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/neobot_no_mass_matrix/urdf/neobot_no_mass_matrix.urdf'
        ##file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Alexbot_simple_urdf/urdf/Alexbot_simple_urdf.urdf'
        name = "neo"
        foot_name = "ankle_pitch"
        knee_name = "knee"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.#超过此距离将更新地面
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6#0.6#0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05#0.05
            dof_vel = 0.5#0.5
            ang_vel = 0.1#0.1
            lin_vel = 0.05#0.05#0.05
            quat = 0.03#0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.65]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'l_hip_roll_joint': 0.,
            'l_hip_yaw_joint': 0.,
            'l_hip_pitch_joint': -0.,
            'l_knee_pitch_joint': 0.,
            'l_ankle_pitch_joint': 0.,
            
            'r_hip_roll_joint': 0.,
            'r_hip_yaw_joint': 0.,
            'r_hip_pitch_joint': -0.,
            'r_knee_pitch_joint': 0.,
            'r_ankle_pitch_joint': 0.,
                                }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        
        stiffness = {'hip_roll': 50, 'hip_pitch': 100, 'hip_yaw': 50,
                    'knee': 100, 'ankle': 15}
        damping = {'hip_roll': 2, 'hip_pitch': 5, 'hip_yaw': 2,
                'knee': 5, 'ankle': 1}


        # actionscale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25#0.25#0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10#十线程
            solver_type = 1  # 0: pgs, 1: tgs。0代表PGS（Projected Gauss-Seidel），用于快速但可能不太准确的求解；1代表TGS（Temporal Gauss-Seidel），用于更准确但可能稍慢的求解
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]这意味着物理引擎会在物体之间增加一个 0.01 米的额外空间来进行接触检测，从而减少碰撞检测中的错误
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1#0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2]#0.1-2
        randomize_base_mass = True
        added_mass_range = [-0.2, 0.2]#-0.2, 0.2
        push_robots = True
        push_interval_s = 4#每隔四轮推一下机器人
        max_push_vel_xy = 0.2#0.4#0.2
        max_push_ang_vel = 0.4#0.6#0.4
        # dynamic randomization
        dynamic_randomization = 0.02
        action_delay = 0.5
        action_noise = 0.02
        # PD gain randomization
        randomize_pd_gains = False
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2] 

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x =[-0.3, 0.3]   #  [-0.5, 0.5]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   #  [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # 【-0.3, 0.3】  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.6
        min_dist = 0.2
        max_dist = 0.4
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.25#0.25 0.17   # rad
        target_feet_height = 0.07      # m
        cycle_time = 0.64#0.64#0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5#5
        max_contact_force = 300  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos =1.6#1.6#1.6 当前角度与参考动作角度的差距的权重
            feet_clearance = 1#1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1#1.
            foot_slip = -0.05#-0.05
            feet_distance = 0.2
            knee_distance = 0.2

            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2#1.2
            tracking_ang_vel = 1.2#1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2#0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5#0.5
            orientation = 1#1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002#-0.002
            torques = -2e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class neoCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 80  # per iteration
        max_iterations = 1000  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'neo_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
