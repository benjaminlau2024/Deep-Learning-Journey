import torch
from PIL import Image

# 分辨率
H, W = 1080, 1920
# 天青色 RGB：R=128, G=217, B=236
r_val, g_val, b_val = 128, 217, 236

# 1. 创建全0黑色基底 3维张量 [高,宽,通道]
sky_tensor = torch.zeros((H, W, 3))

# 2. 分别给三个通道赋值对应颜色数值
sky_tensor[:, :, 0] = r_val  # R通道全部填128
sky_tensor[:, :, 1] = g_val  # G通道全部填217
sky_tensor[:, :, 2] = b_val  # B通道全部填236

# 转为图片整数格式并保存
sky_img = Image.fromarray(sky_tensor.to(torch.uint8).numpy())
sky_img.save("sky_cyan_1080p.png")

print("天青色图片生成完成 sky_cyan_1080p.png")
print("张量维度：", sky_tensor.shape)