# 🎨 Color Extract — 图片主色/调色盘提取工具

基于 Rust 的高性能图片颜色提取与可视化 demo，包含 **4 个独立 crate**：

| Crate | 用途 |
|-------|------|
| **`dct-extract`** | DCT 增强颜色提取 —— 结合 DCT 纹理复杂度和空间信息的高级评分系统 |
| **`lab-gradient`** | LAB 空间 Sobel 梯度可视化 —— 将三通道梯度映射到 RGB 单图 |
| **`dct-viz`** | DCT 纹理复杂度可视化 —— 8×8 块 DCT 高频能量占比热力图 + 原图叠加 |
| **`spectral-residual`** | 频谱残差显著性检测 —— 2D FFT 频域分析 + 热力图/叠加图 |
| **`color-extract` (根)** | 经典调色盘提取 —— 4 种算法 × 5 个色彩空间并行对比 |

---

# `dct-viz` — DCT 纹理复杂度可视化

## 功能

- 加载 `imgs/` 下的图片，**Lanczos3 缩放至 ≤1024×1024**（保持比例）
- 转灰度后对每个像素取 8×8 邻域做 **Type-II DCT**（与 `dct-extract` 实现一致）
- 计算 **高频能量占比** `c = Σ_{u+v≥4} F(u,v)² / Σ_{all except DC} F²`，取值范围 [0,1]
- 每张图输出 **2 种可视化**：
  | 文件 | 说明 |
  |------|------|
  | `{name}_dct_heat.png` | 彩色热力图：蓝(平滑) → 青 → 黄 → 红(纹理丰富) |
  | `{name}_dct_overlay.png` | 原图灰度叠加：平滑区≈原图灰，纹理区亮绿/青色高亮 |
- **Rayon 多线程**并行计算每个像素的 DCT

## 运行

```bash
cargo run --release -p dct-viz
```

输出在 `output/dct_viz/{filename}_dct_heat.png` 和 `*_dct_overlay.png`。

## 用途

- 直观查看图片的**纹理分布**：平滑区域（低复杂度）vs 细节/边缘区域（高复杂度）
- 辅助理解 DCT 复杂度在颜色提取中的作用——背景往往是平滑区（c 低），主体纹理区（c 高）
- 与 `lab-gradient` 的输出对比：DCT 反映的是**频域**纹理，Sobel 反映的是**空域**梯度

---

# `spectral-residual` — 频谱残差显著性检测

## 功能

- 加载 `imgs/` 下的图片，**Lanczos3 缩放至 ≤1024×1024**（保持比例）
- 转换到 **CIELAB** 色彩空间，提取 L\*、a\*、b\* 三通道
- **对每个通道独立做 2D FFT**（行‑列分离，`rustfft`），计算频谱残差显著性
- **L₂ 范数融合** `S = √(S_L² + S_a² + S_b²)`，再归一化到 [0,1]
- 每张图输出 **2 种可视化**：
  | 文件 | 说明 |
  |------|------|
  | `{name}_sr_heat.png` | 灰度显著图：白=显著，黑=不显著 |
  | `{name}_sr_overlay.png` | 原图 L\* 灰度叠加，显著区域红色高亮 |

## 运行

```bash
cargo run --release -p spectral-residual
```

输出在 `output/spectral_residual/`。

## 原理

频谱残差（Spectral Residual）由 Hou & Zhang (2007) 提出，基于频域信息冗余抑制的思想：
- 自然图像的 log 幅度谱在频域近似平滑 → **平均幅度谱** 表示频域冗余
- 减去均值后的残差就是图片中 **「出乎意料」的频率成分**
- 结合原始相位 IFFT 后，对应空域中的 **显著性区域**
- 本实现使用 **CIELAB 三通道分别计算后 L₂ 融合**，对动漫/插画图的色彩和亮度显著性响应更均衡
- 与 DCT 复杂度不同：DCT 衡量**局部纹理丰富度**，频谱残差检测**全局视觉突出**

---

# `dct-extract` — DCT 增强颜色提取

## 功能

- 对图片做 **8×8 DCT** 运算，提取**局部纹理复杂度**（高频能量占比）
- 结合复杂度、占空比、**CIELAB 色彩特征**，用 **KMeans++ / Mini-Batch KMeans** 聚类
- **三组对比实验**：
  - Baseline：纯 3D (L\*a\*b\*) 聚类
  - 4D (L\*a\*b\* + c)：复杂度维度放大 20× 参与聚类
  - 6D (L\*a\*b\* + c + xy)：再加空间坐标，放大 20×/10×
- **高级评分系统**（七项指标加权，全参数开放调优）：
  ```
  score = p^0.5 · (1+2.7·c_final) · (1+0.20·C_rel) · (1+0.20·L_prom) · (1+0.65·U) · bg_penalty · white_gate
  ```
  - **p**：占比幂次（γ=0.5，开根号抑制大面积背景）
  - **c_final**：DCT 纹理复杂度（绝对截断 + 百分位排名混合）
  - **C_rel**：相对彩度（median + MAD，>0 才奖，偏淡图里相对鲜艳也能加分）
  - **L_prom**：亮度突出度（局部 Gaussian 加权亮度反差 + 背景亮度反差，双边 |ΔL| 对比）
  - **U**：颜色独特性（加权 pairwise ΔE 归一化）
  - **bg_penalty**：背景惩罚（BFS 边界连通 + 边缘 + 散布 + 平滑度）
  - **white_gate**：白/浅灰背景专用门控（chroma<5, L>85, con>0.4 → ×0.15）
- **BFS 边界连通性**检测：从图像四边多源 BFS 识别背景色
- **全参数调参区**：文件顶部集中标注，改完即跑
- 输出自包含 HTML 报告，暗色主题，含诊断徽章（c, C, U, L_prom, B, score）
- 自定义最大尺寸、输出路径

## 运行

```bash
cargo run --release -p dct-extract
cargo run --release -p dct-extract -- <max_dim> <output_path>
```

输出默认位于 `output/results-dct.html`。

## 评分参数一览

调参区位于 `dct-extract/src/cluster_4d.rs` 顶部，所有常量加注释：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `SCORE_GAMMA` | 0.5 | 占比幂次 |
| `SCORE_ALPHA_C` | 2.7 | 复杂度权重 |
| `SCORE_BETA_C` | 0.20 | 相对彩度权重 |
| `SCORE_BETA_L` | 0.20 | 亮度突出度权重（局部+背景反差） |
| `SCORE_BETA_U` | 0.65 | 独特性权重 |
| `SCORE_LAMBDA_B` | 1.0 | 背景惩罚力度 |
| `SCORE_WHITE_GATE` | 0.15 | 白背景门控乘数 |
| `BG_W1~W4` | 0.40/0.30/0.15/0.15 | 背景性 B 的组成 |
| `L_LOCAL_W` / `L_BG_W` | 0.5 / 0.5 | 亮度突出度组成配比 |

---

# `lab-gradient` — LAB 空间 Sobel 梯度可视化

## 功能

- 加载 `imgs/` 下的图片，**Lanczos3 缩放至 ≤1024×1024**（保持比例）
- 转换到 **CIELAB** 色彩空间
- 对 L\*、a\*、b\* **三个通道分别计算 Sobel 梯度幅值**
- **合并为一张 RGB 图输出**：
  - **R 通道** = L\* 梯度幅值（亮度变化）
  - **G 通道** = a\* 梯度幅值（绿-红变化）
  - **B 通道** = b\* 梯度幅值（蓝-黄变化）
- 每个通道独立归一化到 0-255，用最大-最小拉伸保证对比度

## 运行

```bash
cargo run --release -p lab-gradient
```

输出在 `output/lab_gradient/{filename}_grad.png`。

## 用途

- 直观查看 LAB 空间中各维度的边缘/纹理响应
- 分析色彩过渡区域（a\* 和 b\* 梯度强处往往是色相变化处）
- 辅助理解 DCT 复杂度特征与梯度幅值的关系

---

# `color-extract` — 经典调色盘提取

## 功能

- 从图片中提取 **10 色调色盘** + **1 个主色**
- 同时对比 **4 种颜色量化算法**：
  | 算法 | 说明 |
  |------|------|
  | **KMeans++** | 基于 `linfa-clustering`，KMeans++ 初始化聚类 |
  | **Mini-Batch KMeans** | 增量式 KMeans，batch=2048 |
  | **Median Cut** | 中位切分量化，自定义实现 |
  | **Octree** | 八叉树量化（RGB 用位索引，其他空间用范围索引） |
- 每种算法在 **5 个色彩空间** 中分别运行：
  | 色彩空间 | 说明 |
  |----------|------|
  | **RGB** | 原始 sRGB 空间 |
  | **CIELAB** | CIE L\*a\*b\*，感知均匀 |
  | **Oklab** | 现代感知均匀色彩空间 |
  | **HSL** | 色相-饱和度-明度 |
  | **CAM16** | CIE 色彩外观模型 (CAM16-UCS) |
- **多线程加速** (rayon)，3 张图 × 20 种组合 = 60 个任务并行执行
- 输入图片自动缩放至 ≤1024×1024
- 调色盘按 **CIELAB L\*** 从暗到亮排序
- 主色 **不从调色盘中取**，而是使用算法内部最具代表性的颜色
- 记录每种组合的 **运行耗时**
- 输出 **自包含 HTML 文件**，暗色主题，可直接在浏览器中查看

## 运行

```bash
cargo run --release
cargo run --release -- <max_dim> <output_path>
# 例如：
cargo run --release -- 512 output/my-results.html
```

输出默认位于 `output/results.html`。

---

## 工作区结构

```
color-extract/
├── Cargo.toml                     # 工作区根
├── dct-extract/                   # DCT 增强颜色提取
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                # CLI 入口
│       ├── img.rs                 # 图片加载/缩放
│       ├── dct.rs                 # 8×8 DCT 实现
│       ├── cluster_4d.rs          # 聚类 + 高级评分系统（调参区）
│       └── html.rs                # HTML 报告生成
├── dct-viz/                       # DCT 纹理复杂度可视化
│   ├── Cargo.toml
│   └── src/
│       └── main.rs                # 加载→缩放→DCT→热力图+叠加图输出
├── spectral-residual/            # 频谱残差显著性检测
│   ├── Cargo.toml
│   └── src/
│       └── main.rs                # 加载→缩放→FFT→频谱残差→IFFT→热力图+叠加图
├── lab-gradient/                  # LAB Sobel 梯度可视化
│   ├── Cargo.toml
│   └── src/
│       └── main.rs                # 加载→缩放→LAB→Sobel→RGB 输出
├── src/                           # 经典调色盘提取
│   ├── main.rs
│   ├── img.rs
│   ├── colorspace.rs              # 5 个色彩空间转换 + CAM16-UCS 反算
│   ├── html.rs
│   ├── timing.rs
│   └── algorithms/
│       ├── mod.rs
│       ├── kmeans.rs
│       ├── minibatch_kmeans.rs
│       ├── median_cut.rs
│       └── octree.rs
├── imgs/                          # 输入图片目录
├── output/                        # 生成的 HTML 报告 & 可视化图
│   ├── results.html               # 经典调色盘报告
│   ├── results-dct.html           # DCT 增强报告
│   ├── lab_gradient/              # LAB 梯度可视化
│   ├── dct_viz/                   # DCT 复杂度可视化
│   └── spectral_residual/         # 频谱残差显著性图
└── README.md
```

## 性能

### dct-extract
| 指标 | 数值 |
|------|------|
| 测试图数 | 37 张动漫插画/照片 |
| 总耗时 | ~90-125s (release, 含 6 种聚类方式) |
| 4D KMeans++ | ~0.3-1.7s/图 |
| 单图聚类数 | 10 |

### root color-extract (经典版)
| 指标 | 数值 |
|------|------|
| 总组合数 | 60 (3图 × 4算法 × 5空间) |
| 总耗时 | ~5s (release) |
| 最快算法 | Median Cut (~170ms/组合) |
| 最慢算法 | KMeans++ (~2-4s/组合) |

## 依赖

| Crate | 用途 |
|-------|------|
| `rustfft` 6 | 2D FFT 频谱残差计算 |
| `image` 0.25 | 图片加载/缩放 |
| `palette` 0.7 | CIELAB、Oklab、HSL、CAM16 等色彩空间 |
| `linfa-clustering` 0.8 | KMeans++ / Mini-Batch KMeans |
| `ndarray` 0.16 | 矩阵计算 |
| `rayon` 1 | 多线程并行 |
| `base64` 0.22 | 图片内嵌 HTML |
| `chrono` 0.4 | 时间戳 |
| `rand_xoshiro` 0.6 | 确定性随机数种子 |
| `anyhow` 1 | 错误处理 |

## 构建

```bash
# 需要 Rust 1.84+ (edition 2024)
cargo build --release
```

## 技术细节

### CAM16 色彩外观模型

通过 `palette` crate 实现，使用 sRGB 标准观看条件（40 cd/m²）。CAM16-UCS J'a'b' 坐标的反向转换使用 Li et al. (2017) 的数学公式手动实现，绕过了 palette 的类型系统限制。

### 八叉树双策略

- **RGB 空间**：经典位索引八叉树，直接对 0-255 整数坐标按位分组
- **其他空间**：范围索引八叉树，在每个节点计算坐标范围的二分中点来分组

### 主色提取（经典版）

主色来自算法内部状态，而非调色盘排序结果：
- KMeans/Mini-Batch：像素数最多的聚类的质心
- Median Cut：最大 bucket 的均值
- Octree：像素数最多的叶子节点的均值

### 频谱残差显著性

频谱残差法（Spectral Residual, Hou & Zhang 2007）通过 2D FFT 将图片变换到频域，提取 log 幅度谱并减去 3×3 均值滤波后的平均幅度谱，残差部分代表频域中的非冗余信息。结合原始相位经 IFFT 重建后得到空域显著性图。与 DCT 复杂度不同的是：DCT 度量局部 8×8 块的纹理丰富度，频谱残差检测全局视觉突出区域。

### DCT 复杂度

8×8 块 Type-II DCT 后，高频系数绝对值之和与总系数绝对值之和的比值作为块级复杂度。聚类时复杂度放大 20× 参与距离计算，在评分环节再除以 20 归一化。

### 亮度突出度

使用**局部 Gaussian 加权亮度反差** + **背景亮度反差**的双边对比度机制，替代旧的彩度联动方案。亮色在暗图中、暗色在亮图中均能获得亮度加分，真正反映"突出"而不仅是"亮"。

## License

MIT
