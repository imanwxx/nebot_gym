import os
import sys
import numpy as np
import itertools
import math 
import time

sys.path.append("/opt/openrobots/lib/python3.8/site-packages")

from pinocchio import visualize
import pinocchio
# import example_robot_data
import crocoddyl
from pinocchio.robot_wrapper import RobotWrapper
 
current_directory = os.getcwd()
print("上层路径：", current_directory)
 
# change path ??
# modelPath = current_directory + '/resources/robots/XBot/'
# URDF_FILENAME = "urdf/XBot-L.urdf"
modelPath=current_directory
URDF_FILENAME="/neobot_no_mass_matrix.urdf"
# modelPath=current_directory+'/resources/robots/jvrc/urdf'
# URDF_FILENAME="/jvrc.urdf"
 
# Load the full model
rrobot = RobotWrapper.BuildFromURDF(modelPath + URDF_FILENAME, [modelPath], pinocchio.JointModelFreeFlyer())  # Load URDF file
rmodel = rrobot.model
 
rightFoot = 'r_ankle_pitch_link'
leftFoot = 'l_ankle_pitch_link'
 
display = crocoddyl.MeshcatDisplay(
    rrobot, frameNames=[rightFoot, leftFoot]
)
q0 = pinocchio.utils.zero(rrobot.model.nq)

print("q0:", q0)
display.display([q0])
time.sleep(3)
print("---------------initial pos-----------")
 
rdata = rmodel.createData()
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
 
rfId = rmodel.getFrameId(rightFoot)
lfId = rmodel.getFrameId(leftFoot)
 
rfFootPos0 = rdata.oMf[rfId].translation
lfFootPos0 = rdata.oMf[lfId].translation
 
comRef = pinocchio.centerOfMass(rmodel, rdata, q0)
 


# 定义目标角度和当前角度
# target_angle = 3.14  # 目标角度（弧度）
# current_angle = q0[7]  # 当前角度

# # 设置运动步数和时间间隔
# steps = 100
# time_interval = 0.1  # 每步持续的时间（秒）

# # 逐步更新关节角度
# for step in range(steps + 1):
#     # 线性插值计算当前关节角度
#     q_current = q0.copy()
#     q_current[7] = current_angle + (target_angle - current_angle) * (step / steps)

#     # 更新机器人的位置
#     pinocchio.forwardKinematics(rmodel, rdata, q_current)
#     pinocchio.updateFramePlacements(rmodel, rdata)

#     # 可视化更新
#     display.display([q_current])
    
#     # 等待指定的时间间隔
#     time.sleep(time_interval)

 
# print("--------------start to play ref pos--------------")
for i in range(1000):
    phase = i * 0.005
    sin_pos = np.sin(2 * np.pi * phase)
    sin_pos_l = sin_pos.copy()
    sin_pos_r = sin_pos.copy()
 
    ref_dof_pos = np.zeros((1,10))
    scale_1 = 0.25
    scale_2 = 2*scale_1
    scale_3 = scale_1
    # left foot stance phase set to default joint pos
    if sin_pos_l > 0 :
        sin_pos_l = sin_pos_l * 0
    ref_dof_pos[:, 2] = -sin_pos_l * scale_1#hip_pitch
    ref_dof_pos[:, 3] = sin_pos_l * scale_2#knee_pitch
    ref_dof_pos[:, 4] = sin_pos_l * scale_3#ankle_pitch
    # right foot stance phase set to default joint pos
    if sin_pos_r < 0:
        sin_pos_r = sin_pos_r * 0
    ref_dof_pos[:, 7] = sin_pos_r * scale_1
    ref_dof_pos[:, 8] = -sin_pos_r * scale_2
    ref_dof_pos[:, 9] = sin_pos_r * scale_3
    # Double support phase
    ref_dof_pos[np.abs(sin_pos) < 0.1] = 0
 
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[7:rrobot.model.nq] = ref_dof_pos
    display.display([q0])
    time.sleep(0.1)
print("-------------finish-----------------")
 
 
 
 
 
# for i in range(rrobot.model.nq-7):
#     q0 = pinocchio.utils.zero(rrobot.model.nq)
#     q0[6] = 1  # q.w
#     q0[2] = 0  # z
#     q0[i+7] = 1
#     display.display([q0])
#     print("------------reset pos-----------------")

