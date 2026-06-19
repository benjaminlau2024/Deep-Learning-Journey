# Chapter 01: 张量基础与图像生成 (Tensors & Basics)

## 📌 学习导读
在进入复杂的神经网络之前，我们首先要搞清楚深度学习处理数据的基本单位——**张量 (Tensor)**。
对于有视频后期背景的 **Benjamin** 来说，理解张量有一个绝佳的物理模型：**视频的“帧 (Frame)”**。
* 一个 1080P 的视频帧，本质上就是一个包含 `1080像素(高) × 1920像素(宽) × 3通道(RGB)` 的三维矩阵。
* 在 PyTorch 中，这个三维矩阵就是**三维张量 (3D Tensor)**。

---

## 📂 练习代码与实验说明

在本章中，我们通过 4 个 Python 脚本，分别以不同的张量初始化和切片赋值方式，生成了不同的图片：

### 1. 📂 [whitepaper.py](file:///d:/Deep-Learning-Journey/01_Tensors_and_Basics/whitepaper.py) —— 生成纯白壁纸
* **核心方法**：使用 `torch.ones((H, W, C))` 创建一个全 1 的三维张量，然后乘以 `255`。
* **物理意义**：RGB 通道全满 `[255, 255, 255]` 即为白色。

### 2. 📂 [redpaper.py](file:///d:/Deep-Learning-Journey/01_Tensors_and_Basics/redpaper.py) —— 生成纯红壁纸
* **核心方法**：使用 `torch.zeros((H, W, C))` 创建全黑底色，然后通过张量切片对第 0 通道（红通道）整体赋值 `255`：
  ```python
  red_tensor[:, :, 0] = 255
  ```
* **物理意义**：只点亮红光，绿光与蓝光全灭。

### 3. 📂 [celestepaper.py](file:///d:/Deep-Learning-Journey/01_Tensors_and_Basics/celestepaper.py) —— 生成天青色壁纸
* **核心方法**：定义天青色的 RGB 数值 `[128, 217, 236]`，然后通过切片分别对 3 个通道赋值：
  ```python
  sky_tensor[:, :, 0] = 128  # R
  sky_tensor[:, :, 1] = 217  # G
  sky_tensor[:, :, 2] = 236  # B
  ```

### 4. 📂 [randompaper.py](file:///d:/Deep-Learning-Journey/01_Tensors_and_Basics/randompaper.py) —— 生成彩色随机噪点图
* **核心方法**：利用 `torch.randint(0, 256, size=(H, W, 3), dtype=torch.uint8)` 在每个像素的每个通道上生成 0~255 的随机整数。
* **物理意义**：创造出完全杂乱无章的电视雪花噪点。

---

## 💡 Helen 的深度学习小课堂（重点笔记）

> [!TIP]
> **1. uint8 与 float32 的转换**
> * **uint8（无符号 8 位整数）**：取值范围为 `0 ~ 255`。这是图像文件（PNG/JPG）的标准存储格式，也是 PIL 库保存图像时所必需的。
> * **float32（单精度浮点数）**：取值范围为 `0.0 ~ 1.0`（归一化之后）。在后续的神经网络训练中，所有的张量都需要被转换成浮点数（并且除以 255 归一化）。因为浮点数支持微小的梯度更新，神经网络才能通过梯度下降逐步学习！
> * **转换代码**：`tensor.to(torch.uint8)` 或 `tensor.to(torch.float32)`。

> [!IMPORTANT]
> **2. 图像通道顺序的“陷阱” (HWC vs CHW)**
> 在处理图像深度学习时，新手最容易被通道顺序搞懵。请务必记住这两种规范：
> * **HWC（高, 宽, 通道）**：这是 NumPy、PIL 和我们本章保存图片时使用的顺序，也是直观感受图片的顺序（例如 `[1080, 1920, 3]`）。
> * **CHW（通道, 高, 宽）**：这是 **PyTorch 卷积神经网络 (CNN)** 要求的默认格式（如 `[3, 1080, 1920]`）。
> * **如何转换？** 在后面的学习中，我们会使用 PyTorch 的 `.permute(2, 0, 1)`（将 HWC 转为 CHW）或者 `.permute(1, 2, 0)`（转回 HWC 保存图片）。

---

## 🌟 Benjamin 的学习心得打卡
* 💡 **今日所学**：掌握了 PyTorch 的张量切片操作，理解了 RGB 通道在计算机里的张量表达。
* 🎯 **下一步计划**：尝试构建第一个线性回归模型（02_Linear_Regression），看看 PyTorch 是如何帮我们自动计算梯度的！
