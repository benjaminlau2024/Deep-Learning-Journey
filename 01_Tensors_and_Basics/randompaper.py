import torch
from PIL import Image

# 画布尺寸
H, W = 1080, 1920

# 生成随机张量，范围0~255，shape=[H,W,3]
# randint(最小值,最大值,尺寸) 生成整数随机数
random_img_tensor = torch.randint(0, 256, size=(H, W, 3), dtype=torch.uint8)

# 转numpy保存图片
img = Image.fromarray(random_img_tensor.numpy())
img.save("random_color_1080p.png")

print("随机彩色噪点图已生成 random_color_1080p.png")
print("图片张量维度：", random_img_tensor.shape)