import torch
from PIL import Image

# 1. 定义分辨率：高度1080，宽度1920，3通道RGB
H, W, C = 1080, 1920, 3

# 2. 创建全1张量（0~1标准），乘255转为图片0~255数值
white_tensor = torch.ones((H, W, C)) * 255

# 3. 转换格式：张量→整数numpy数组→PIL图片
# permute调换通道顺序是PyTorch标准操作，适配图片保存
white_np = white_tensor.to(torch.uint8).numpy()
white_img = Image.fromarray(white_np)

# 4. 保存图片到本地
white_img.save("white_1080p.png")
print("1920×1080纯白图片已生成：white_1080p.png")

# 可选：打印张量维度验证
print("图片张量shape（高,宽,通道）：", white_tensor.shape)