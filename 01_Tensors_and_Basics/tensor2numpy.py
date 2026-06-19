import torch
# 1. 创建2行2列的3维彩色张量（2×2小图，每个像素RGB）
t = torch.tensor([
    [[255.0, 0.0, 0.0],    # 第0行第0个像素：红色
     [0.0, 255.0, 0.0]],   # 第0行第1个像素：绿色
    [[0.0, 0.0, 255.0],    # 第1行第0个像素：蓝色
     [255.0, 255.0, 255.0]]# 第1行第1个像素：白色
])
print("原始tensor：")
print(t)
print("原始数据类型 dtype：", t.dtype)
print("-"*50)

# 第一步：.to(torch.uint8) 浮点→8位整数
t_uint8 = t.to(torch.uint8)
print("执行 .to(torch.uint8) 之后：")
print(t_uint8)
print("转换后 dtype：", t_uint8.dtype)
print("-"*50)

# 第二步：.numpy() tensor → numpy数组
arr = t_uint8.numpy()
print("执行 .numpy() 得到numpy数组：")
print(arr)
print("数组类型 type(arr)：", type(arr))