import torch
from PIL import Image

# 图片尺寸 高1080 宽1920 3通道RGB
H, W, C = 1080, 1920, 3

# 1. 先创建全0张量，默认全黑 [0,0,0]
red_tensor = torch.zeros((H, W, C))

# 2. 所有像素的第0通道（R红色通道）全部赋值255
red_tensor[:, :, 0] = 255

# 修正点：先转uint8，再转numpy数组
red_uint8 = red_tensor.to(torch.uint8).numpy()
red_img = Image.fromarray(red_uint8)
red_img.save("red_1080p.png")

print("纯红色1080P图片已生成 red_1080p.png")
print("张量尺寸：", red_tensor.shape)