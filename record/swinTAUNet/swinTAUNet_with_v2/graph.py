import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# 设置文件夹路径
directory = r'C:\Users\rindon\Desktop'

# 从指定文件夹读取所有CSV文件
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 设置图形大小
plt.figure(figsize=(14, 10))

for file in files:
    filepath = os.path.join(directory, file)
    data = pd.read_csv(filepath)
    
    # 读取精度和损失数据
    Accuracy = data.iloc[:, 4]
    Epoch = range(1, len(Accuracy) + 1)  # 确保Epoch的范围与数据长度一致

    # 过滤掉Accuracy中的0值
    mask = Accuracy != 0
    filtered_Epoch = [epoch for i, epoch in enumerate(Epoch) if mask.iloc[i]]  # 过滤Epoch以匹配过滤后的Accuracy
    
    plt.plot(filtered_Epoch, Accuracy[mask], label=file[:-4])  # 只绘制非0的Accuracy值

# 添加图例
plt.legend()

# 设置图形标题和坐标轴标签
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Dice', fontsize=15)

# 设置坐标轴范围
plt.xlim((0, len(Epoch)))
plt.ylim((0, 0.75))

# 设置Y轴的刻度间隔
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.02))

# 显示网格
plt.grid()

# 显示图形
plt.show()
