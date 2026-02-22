from datasets import load_dataset

# 加载数据集
dataset = load_dataset("/hdd0/zhongqishu/dataset/math/deepmath-103k")
# 查看数据集的结构
print(dataset)

# 访问训练集
train_dataset = dataset['train']

# 访问测试集
test_dataset = dataset['test']

# 访问第一个样本
first_sample = train_dataset[0]
print(first_sample)
