import pandas as pd
import pickle
# 读取.pkl文件并转换为DataFrame对象
filename = 'test.pkl'
#df = pd.read_pickle(filename)
# 显示数据摘要
#print(df.head())  # 输出前5行数据
#print(df.describe())  # 输出描述性统计信息

f = open(filename, 'rb')
data = pickle.load(f)
print(data)
