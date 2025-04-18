import numpy as np
import matplotlib.pyplot as plt

# 读取两个 CSV 文件中的关节角度数据
joint_angles_1 = np.loadtxt("joint_angles.csv", delimiter=",")
joint_angles_2 = np.loadtxt("joint_angles_play.csv", delimiter=",")

# 找到较小的数据量
min_steps = min(joint_angles_1.shape[0], joint_angles_2.shape[0])

# 定义时间步长和正弦函数，以较小的数据量为标准
time_steps = np.arange(min_steps) * 0.01  # 假设时间步长为 0.01 秒
sine_wave = 0.25 * np.sin(2 * np.pi * time_steps / 0.64)  # 周期为 0.64 秒，幅值为 0.25

# 截取较小数据量的关节角度数据
joint_angles_1 = joint_angles_1[:min_steps, :]
joint_angles_2 = joint_angles_2[:min_steps, :]

# 绘制曲线
plt.figure(figsize=(12, 8))

# 绘制第一个 CSV 文件中的关节角度
for i in range(joint_angles_1.shape[1]):
    if i==7  or i==9:
        # 只绘制第8、9、10个关节角度
    
        plt.plot(time_steps, joint_angles_1[:, i], label=f"Joint {i+1} (CSV 1)")

# 绘制第二个 CSV 文件中的关节角度
for i in range(joint_angles_2.shape[1]):
    if i==7  or i==9:
        # 只绘制第8、9、10个关节角度
        plt.plot(time_steps, joint_angles_2[:, i], linestyle="--", label=f"Joint {i+1} (CSV 2)")

# 绘制正弦函数
plt.plot(time_steps, sine_wave, label="Sine Wave (0.64s period, 0.25 amplitude)", linestyle=":", color="black")

# 图形设置
plt.xlabel("Time (s)")
plt.ylabel("Joint Angles (rad)")
plt.title("Comparison of Joint Angles from Two CSV Files and Sine Wave")
plt.legend()
plt.grid()
plt.savefig("joint_angles_comparison_with_sine_wave.png")
plt.show()