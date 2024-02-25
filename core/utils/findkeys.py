import torch

# 替换这里的路径为你的.pkl文件的路径
file_path = 'models/43_6_2_vonet_30000.pkl'

# 加载.pkl文件
data = torch.load(file_path)

# 打印所有的键
print("Keys in the .pkl file:")
for key in data.keys():
    print(key)
