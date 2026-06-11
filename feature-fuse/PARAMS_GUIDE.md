# feature-fuse 参数配置指南

> 修改后无需重新编译，直接运行即可生效。
> 留空/缺失的字段会使用 serde 默认值。

---

## 目录

1. [图片预处理](#1-图片预处理)
2. [局部 Gaussian 残差](#2-局部-gaussian-残差)
3. [Percentile 归一化](#3-percentile-归一化)
4. 全局残差 & 后处理
   - [4. 全局残差基准值选择](#4-全局残差基准值选择)
   - [4a. 饱和度后处理](#4a-饱和度特征后处理参数)
   - [4b. 背景估计（三阶段管线）](#4b-背景估计参数三阶段管线)
   - [4c. Subject Prior（高斯中心偏向）](#4c-subject-prior高斯中心偏向)
   - [4d. Segment-aware Region Priors（color-segment 区域先验）](#4d-segment-aware-region-priorscolor-segment-区域先验)
5. [特征权重](#5-特征权重)
6. [Hybrid Fusion](#6-hybrid-fusion-参数)
   - [6a. Direct Blend](#6a-direct-blend-参数)
7. [最终过滤](#7-最终-fuse-图过滤)
8. [Contact Sheet](#8-contact-sheet-拼贴图布局)
9. [DCT 纹理复杂度](#9-dct-纹理复杂度参数)
10. [频谱残差显著性](#10-频谱残差显著性参数)
11. [印象色聚类](#11-印象色聚类参数)
12. [动态特征权重](#12-动态特征权重dynamic-feature-weights)

---

## 1. 图片预处理

**`max_dim`** — 图片最长边限制（像素）。

输入图若长边 > max_dim，按 Lanczos3 等比缩放到 max_dim。

| 方向 | 效果 |
|------|------|
| 调小 | 速度更快、特征更粗糙 |
| 调大 | 细节保留更多、内存+时间更大 |

**建议范围:** 360~1280，默认 720 在速度和精度间平衡较好。

---

## 2. 局部 Gaussian 残差

**`gauss_sigma`** — Gaussian 模糊的 sigma（标准差），用于 `local_light` / `local_lab_a` / `local_lab_b` 残差计算。

**原理:** `|原图 - Gaussian模糊图|`，模糊后丢失的细节即为"局部残差"。

| 方向 | 效果 |
|------|------|
| sigma 越大 | 模糊越强 → 残差捕获更大尺度的局部变化（粗纹理/色块） |
| sigma 越小 | 模糊越弱 → 残差只捕获细微边缘（细纹理/噪点） |

**建议范围:** 5~80，默认 35.0 适合中等尺度的局部细节。

---

## 3. Percentile 归一化

**所有特征图共用。** 每张特征图计算完成后，做 percentile 截断 + 线性映射到 [0, 1]。

这样做可以消除离群值对归一化的影响，让特征图对比度更稳定。

### 参数

| 参数 | 默认 | 效果 |
|------|------|------|
| `percentile.low` | 1.0 | 低于此百分位的像素值 → 置 0（去除暗部离群值） |
| `percentile.high` | 99.0 | 高于此百分位的像素值 → 置 1（去除亮部离群值） |

### 调参说明

| 调参 | 效果 |
|------|------|
| low 调大 (如 5.0) | 压制更多暗区噪声，特征图暗部更"干净"，但可能丢失弱信号 |
| high 调小 (如 95.0) | 增强亮部对比度，让高响应区域更突出，但可能过曝 |
| 极端 low=0 / high=100 | 等价于 min-max 归一化（易受离群值干扰） |

**建议范围:** low 0~5, high 95~100，默认 (1, 99) 对大多数图效果稳健。

---

## 4. 全局残差基准值选择

**稳健中心估计** — `light` / `lab_a` / `lab_b` 各有一套独立的稳健中心参数，结构一致。

### 流程

1. 感知压缩 (gamma 或 log)
2. 压缩域 clip 到 [p{trim_low}, p{trim_high}]
3. trimmed_mean
4. 混合: `trimmed_mean_weight × trimmed_mean + median_weight × median`
5. 从压缩域还原到 [0, 1]
6. 残差 = `|像素值 - center|`

### 必要字段

| 字段 | 说明 |
|------|------|
| `compression` | `"gamma"` — pow(gamma_power) \| `"log"` — log_base(1+eps) 归一化 |
| `trim_low` / `trim_high` | 压缩域截断百分位，剔除离群值 |

### 可选字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `trimmed_mean_weight` | 0.65 | 混合系数 |
| `median_weight` | 0.35 | 混合系数 |
| `gamma_power` | 0.5 | gamma 压缩指数 |
| `log_base` | e | log 底数（仅 compression=log 时有效） |

### 正确配置示例

```yaml
light:
  compression: gamma
  trim_low: 2.0
  trim_high: 98.0
  trimmed_mean_weight: 0.7
  median_weight: 0.3
  gamma_power: 0.4
```

### 通道说明

| 通道 | 说明 |
|------|------|
| `light` | L* 明度全局残差 |
| `lab_a` | Lab a\* 红-绿轴全局残差。a\* ≈ [-128, 127]，计算时自动归一化到 [0,1] |
| `lab_b` | Lab b\* 黄-蓝轴全局残差。b\* ≈ [-128, 127]，计算时自动归一化到 [0,1] |
| `sat` | HSL 饱和度全局残差。hsl_s 已是 [0, 1]，直接做稳健中心残差 |

---

## 4a. 饱和度特征后处理参数

专用于本地饱和度残差的 post-gamma 压缩。

**`saturation.local_post_gamma`** — 局部饱和度残差后处理指数。

在 Gaussian 残差计算后、percentile 归一化前: `sat = raw^gamma`。

| 值 | 效果 |
|----|------|
| <1 (如 0.5) | 抬升低饱和度差异 → 更多"微弱色彩变化"被响应 |
| =1 | 无变化 |
| >1 (如 2.0) | 压制低值 → 只有高饱和度差异幸存 |

**建议范围:** 0.3~3.0，默认 1.0。

---

## 4b. 背景估计参数（三阶段管线）

### 管线概览

**Phase 1 — Median Cut 色域切分:**
将像素在 LAB 空间递归二分，每层选方差最大通道的 median 切分。
停止条件: depth ≥ max_depth 或总方差 < variance_threshold。
合并小簇后缩减至 ≤ max_clusters。对每个簇按边界占比 + 色调距离评分，阈值筛选为背景候选。

**Phase 2 — BFS 连通 + 前景结构阻断 + 形态学:**
从边界背景候选种子出发做 4-邻域 BFS 区域生长。扩张必须同时满足颜色背景似然、LAB 距离和非 barrier，避免连接到边界的发丝、披帛、衣物纹理被背景 flood 吞掉。再做闭→开→腐蚀，填充孔洞、去除噪声。max_bg_ratio 保护防止过度检测。

**Phase 3 — 软 mask 精炼:**
从边界采样最多 96 个原型色，逐像素计算高斯距离相似度 → 软背景似然图。根据硬 mask 的统计特性自适应混合硬/软 mask。硬 mask 退化为常量时，BGM-Mask 会回退为软前景置信度。

### 输出特征图（均为前景导向，高值=前景）

| 特征图 | 说明 |
|--------|------|
| `background_mask_morph` | 前景导向 mask，硬分区弱混合软前景置信度以保留边缘 |
| `background_fg_confidence` | 软置信度（值域 [0,1]），经 box blur 平滑 |

前景置信度 = 反背景似然 ×0.45 + 中心偏向 ×0.40 + 显著性占位 ×0.15。

---

### Phase 1: Color Partition (Median Cut)

#### `partition.enabled` (默认 true)

是否启用色域切分做背景估计。

| 值 | 效果 |
|----|------|
| false | 两个输出特征图全为 1.0（所有像素视为前景），背景相关权重退化为常数乘以 1.0。适用于主体撑满画面、无法从边界推断背景的场景 |

#### `partition.max_depth` (默认 5, 建议 4~8)

Median Cut 最大递归深度。每深一层，当前簇沿 LAB 方差最大通道的 median 一分为二。最大叶簇数 = 2^max_depth（depth=5 → 最多 32 簇，合并前）。

| 方向 | 效果 |
|------|------|
| 调大 | 更细的颜色分区，捕获微秒色调差异 |
| 调小 | 更粗的分区，减少后续合并开销 |

#### `partition.max_clusters` (默认 16, 建议 6~24)

合并小簇后若簇数仍超此值，反复将最小簇合并到 LAB 距离最近的大簇，直至 ≤ 上限。

| 方向 | 效果 |
|------|------|
| 调大 | 保留更多颜色簇进入评分阶段，可能引入噪声 |
| 调小 | 合并更激进，可能把不同的颜色混在一起，降低区分能力 |

#### `partition.variance_threshold` (默认 0.1, 建议 0.03~0.3)

LAB 三通道方差之和（var_L + var_a + var_b）低于此值则停止切分。值域未经归一化，所以阈值依赖 LAB 数值大小（L 0~100, a/b 约 ±128）。

| 方向 | 效果 |
|------|------|
| 调大 | 更早停止切分（更多区域被判为"已足够均匀"），适合色块分明的插画/设计图 |
| 调小 | 继续切分直至区分出微妙颜色渐变，适合自然摄影 |

#### `partition.min_cluster_area_ratio` (默认 0.01, 建议 0.0~0.05)

像素数低于 total_pixels × ratio 的簇视为"小簇"，合并到 LAB 距离最近的大簇。

| 值 | 效果 |
|----|------|
| ratio=0.01 | 小于 1% 图像面积的簇被清除 |
| 0 | 不合并（保留所有微小簇） |
| 调大 | 更激进地清除噪点色块，但也可能吞掉小而重要的颜色区域 |

#### `partition.border_band` (默认 3, 建议 2~20)

图像四周的采样条带宽度（像素）。同时用于:

1. 采样边界像素计算背景 LAB 原型色（为每个簇的色调距离提供参照）
2. BFS 种子点选取（只有条带内且 mask ≥ bg_score_threshold 的像素可做种子）
3. Phase 3 软 mask 的原型色采样（最多 96 个边界像素）

| 方向 | 效果 |
|------|------|
| 调大 | 采样更多边界像素，对边角有非典型颜色的图像更鲁棒，但可能包含边缘主体像素 |
| 调小 | 更严格的边界定义，只在最外缘采样 |

#### `partition.bg_score_threshold` (默认 0.55, 建议 0.35~0.75)

背景评分阈值。簇评分 = border_ratio × 0.6 + (1 - center_dist) × 0.4，其中:

- `border_ratio` = 簇中位于边界 band 内的像素占比
- `center_dist` = 簇均值 LAB 到边界 LAB 原型色的归一化距离

评分 ≥ 阈值 → 该簇归为背景 (mask=1)。

| 方向 | 效果 |
|------|------|
| 调大 | 更保守（只有强烈边界+色调证据的簇才判为背景），主体更完整但背景残留更多 |
| 调小 | 更激进（中低证据的簇也判为背景），背景更干净但可能误伤主体 |

#### `partition.bg_connect_threshold` (默认 0.08, 建议 0.03~0.15)

BFS 区域生长的 LAB 距离阈值。对邻域像素计算:

`dist = sqrt((ΔL/100)² + (Δa/128)² + (Δb/128)²)`

dist ≤ 阈值 → 加入连通背景。

| 方向 | 效果 |
|------|------|
| 调大 | BFS 扩散到更多颜色 → 更多像素被划为背景 |
| 调小 | 只在极相似颜色间传播 |

#### `partition.max_bg_ratio` (默认 0.85, 建议 0.5~0.95)

最大背景占比安全阀。两次检查分别在 BFS 后和 morph 后: 若 mask_mean > ratio，放弃 BFS/morph 结果，回退到原始簇 mask。

| 方向 | 效果 |
|------|------|
| 调大 | 允许更多像素被划为背景才触发回退 |
| 调小 | 更严格限制背景范围 |

---

### Phase 2: Morphology

#### `morphology.close_radius` (默认 8, 建议 0~15)

闭运算（先膨胀再腐蚀）半径。填充背景 mask 中的小空洞（被背景包围的前景孤岛）。

公式: `closed = erode(dilate(mask, r), r)`

| 方向 | 效果 |
|------|------|
| 调大 | 填充更大空洞，去除非连通的前景噪点 |
| 0 | 跳过闭合 |

#### `morphology.open_radius` (默认 2, 建议 0~5)

开运算（先腐蚀再膨胀）半径。去除背景 mask 中的小前景噪声块。

公式: `opened = dilate(erode(mask, r), r)`

| 方向 | 效果 |
|------|------|
| 调大 | 消除更大的噪声散点 |
| 0 | 跳过开运算 |

#### `morphology.erode_radius` (默认 3, 建议 0~6)

最终腐蚀半径。腐蚀背景 mask → 缩小背景区域 → 等效于扩展前景。

| 方向 | 效果 |
|------|------|
| 调大 | 更激进地扩展前景 |
| 0 | 跳过最终腐蚀 |

---

### Phase 2b: Flood Barrier / Foreground Protect

#### `flood_barrier.enabled` (默认 true)

是否启用背景 flood 的前景结构阻断。关闭后退回旧式 LAB 连通扩张。

#### `flood_barrier.color_threshold` (默认 0.45)

背景颜色似然阈值。BFS 只能扩张到 border_bg > threshold 的像素。

| 方向 | 效果 |
|------|------|
| 调大 | 背景扩张更保守 |
| 调小 | 平滑浅色背景更容易被连通识别 |

#### `flood_barrier.barrier_color_relax_threshold` (默认 0.92)

高置信背景放行阈值。`border_bg` ≥ 此值时忽略结构 barrier，避免平滑白背景被 DCT/残差归一化噪声误挡。

| 方向 | 效果 |
|------|------|
| 调高 | barrier 更严格 |
| 调低 | 背景更容易连通 |

#### `flood_barrier.*_stop`

barrier 阈值。任一结构图超过阈值即阻断 BFS:

| 参数 | 作用 |
|------|------|
| `grad_stop` | LAB 梯度阻断 |
| `dct_stop` | DCT 纹理阻断 |
| `local_light_stop` | 局部亮度残差阻断 |
| `spectral_stop` | 频谱残差阻断 |

调低 → 更强保护前景细节；调高 → 更少阻断，背景识别更完整。

#### `flood_barrier.protect_strength` (默认 0.75)

morphology 后的软保护强度: `bg_mask *= 1 - strength * foreground_protect`。

| 方向 | 效果 |
|------|------|
| 调大 | 更积极从背景 mask 扣回高纹理/高边缘区域 |
| 调小 | 更保留背景 mask |

#### `flood_barrier.protect_p_low/high` (默认 70/99)

`foreground_protect` 的 percentile normalize 范围。默认只强调结构分数靠前的区域。

#### `flood_barrier.protect_*_weight`

前景保护中各结构特征的混合权重:

| 参数 | 默认 | 作用 |
|------|------|------|
| `protect_grad_weight` | 0.30 | LAB 梯度权重 |
| `protect_dct_weight` | 0.25 | DCT 纹理权重 |
| `protect_spectral_weight` | 0.15 | 频谱残差权重 |
| `protect_local_light_weight` | 0.15 | 局部亮度残差权重 |
| `protect_local_sat_weight` | 0.15 | 局部饱和度残差权重。当前用 `max(local_lab_a_residual, local_lab_b_residual)` 作为代理 |

---

### Phase 3: Soft Mask

#### `soft_mask.border_bg_blur_radius`

边界背景似然图的 box blur 半径。作用于 Phase 3 `border_background_likelihood` 输出。

| 值 | 效果 |
|----|------|
| 0 | 自适应（分辨率相关: `clamp(min(w,h)/48, 4, 16)`，对应 720p 约为 15px） |
| >0 | 固定像素半径。半径越大 → 软背景似然越平滑、边缘过渡越柔和 |

若背景色斑驳可调大 (如 20~30) 使似然图更均匀。

#### `soft_mask.fg_confidence_blur_radius`

前景置信度图的 box blur 半径。作用于 Phase 3 `soft_foreground_from_background` 输出。

| 值 | 效果 |
|----|------|
| 0 | 自适应（分辨率相关: `clamp(min(w,h)/40, 6, 24)`，对应 720p 约为 18px） |
| >0 | 固定像素半径。半径越大 → 前景置信度越平滑、主体轮廓边缘越软 |

若主体边缘有锯齿可调大 (如 30~40) 使过渡更自然。

#### `soft_mask.fg_confidence_sharpen_amount` (默认 0.40)

前景置信度反遮罩锐化强度。作用于 blur 之后、percentile 归一化之前:

`v + (v - blur(v)) * amount`

| 值 | 效果 |
|----|------|
| 0 | 禁用 |
| 0.2~0.5 | 温和提边 |
| 0.8+ | 更硬，也更容易放大噪声或纹理 |

#### `soft_mask.fg_confidence_sharpen_radius` (默认 1)

反遮罩锐化半径。半径越大 → 提升更宽的轮廓过渡；1~3 更适合保留细边缘。

---

## 4c. Subject Prior（高斯中心偏向）

**公式:** `value = exp(-(dx² + dy²))`

其中 `dx = (nx - center_x) / radius_x`，`dy = (ny - center_y) / radius_y`，nx/ny = 归一化像素坐标 [0,1]。

生成一个以 (center_x, center_y) 为中心的 2D 高斯权重图: 中心=1.0，边缘≈0。

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `center_x` | 0.5 | 高斯中心 X（归一化坐标，0~1） |
| `center_y` | 0.55 | 高斯中心 Y，略偏下符合常见构图习惯 |
| `radius_x` | 0.0 | 高斯半径 X（归一化坐标，0~1） |
| `radius_y` | 0.0 | 高斯半径 Y（归一化坐标，0~1） |

### 调参说明

| 方向 | 效果 |
|------|------|
| radius 越大 | 衰减越慢 → 高斯覆盖面越宽，更多像素获得较高先验值 |
| radius 越小 | 衰减越快 → 只有非常靠近中心的像素才有显著先验（更尖锐） |
| radius=0 | 分母趋于 0，除中心像素外所有值 ≈ 0，等效于禁用 subject_prior 特征（整图 ≈0） |

---

## 4d. Segment-aware Region Priors（color-segment 区域先验）

复用 color-segment 的色块分割结果，为背景修正、动态权重和主色提取提供区域级证据。

> 集成路径使用 feature-fuse 已 resize 后的图像，不再按 `segment.preprocess_max_dim` 二次缩放。

### 整体流程

1. **color-segment** 将图像分割为若干连通色块（Union-Find 区域生长）
2. 对每个区域计算 3 个属性：**saliency**（显著度）、**bg_probability**（背景概率）、**subject_confidence**（主体置信度）
3. 区域属性用于 3 个下游：
   - 背景修正（修正 `background_mask_morph` / `background_fg_confidence`）
   - 动态权重（特征加权取决于区域分离能力）
   - 主色提取（直接取 color_score 最高的 Top-1 区域作为主导色）

### 顶层参数

| 参数 | 默认值 | 效果 |
|------|--------|------|
| `enabled` | true | false → 完全禁用区域先验，退回到纯像素级融合 |
| `background_bias` | balanced | "subject" → 加大主体保护（bg_prob 更低）；"background" → 削弱主体保护，背景更易被识别 |
| `diagnostics` | true | true → 输出 segment_*.png 诊断图到 output |

---

### 4d-1. segment — 色块分割参数

传递给 color-segment 库，控制 Union-Find 区域生长算法。

**核心机制:** 逐像素扫描，用 RGB 颜色距离 + LAB Sobel 边缘强度决定是否将当前像素并入相邻区域。生长完成后，通过邻接图合并颜色相近的小区域。

**两大核心杠杆:** `color_weight` 控制颜色敏感度，`edge_weight` 控制边缘阻挡强度。

| 参数 | 默认 | 调大效果 | 调小效果 |
|------|------|----------|----------|
| `min_region_area` | 70 | 更多小区域被吸收 → 分割更粗犷、色块更大 | 保留细小色块 → 分割更精细、碎片更多 |
| `min_cluster_area_ratio` | 0.002 | 小区域阈值 = 面积比例 × 总像素 → 更多吸收。实际阈值 = `max(min_region_area, 比例×总像素)` | — |
| `edge_threshold` | 0.01 | 只有更强梯度才算边缘 → 边缘更稀疏，区域更大 | 几乎全部梯度都算边缘 → 边缘极密，区域极碎。范围 [0, 1]，是梯度最大值的比例 |
| `edge_split_strength` | 0.25 | 边缘墙更高 → 区域更难跨边缘生长 → 更碎。代码内部 clamp [0.25, 0.98] | 边缘墙更低 → 区域更易跨过弱边缘 → 更大块 |
| `edge_gamma` | 0.80 | <1.0（当前值）→ 增强中强度边缘（线稿、渐变边界）；>1.0 → 只有最强边缘保留；=1.0 → 线性 | — |
| `edge_merge_strength` | 0.12 | 允许边界上有较强边缘的区域合并 → 更大区域。合并阶段检查共享边界的平均边缘强度 | 只有边界极弱的区域能合并 → 更保守、更碎 |
| `color_merge_distance` | 8.0 | 颜色差异更大的区域也可以合并 → 更少区域。内部生长用 ×1.75+4，合并用 ×1.9+5 | 颜色必须非常接近才能合并 → 更多区域 |
| `small_region_color_distance` | 20.0 | 小区域即使颜色差异大也容易被吸收 → 更干净 | 小区域颜色差异大时拒绝被吸收 → 保留颜色异常点 |
| `merge_small_regions` | true | true → 吸收面积 < min_region_area 的区域 | false → 保留所有区域，不合并小碎片 |
| `morph_open_radius` | 0 | ↑ 先腐蚀后膨胀，消除孤立的边缘噪点 | 0 → 不做开运算，保留所有边缘 |
| `morph_close_radius` | 1 | ↑ 先膨胀后腐蚀，弥合边缘缺口 → 减少区域泄漏 | 0 → 不做闭运算。1 = 一圈 3×3 dilate+erode |
| `color_weight` | 1.0 | 颜色差异被放大 → 对颜色变化更敏感 → 分割更细。范围 [0.25, 4.0] | 颜色差异被缩小 → 颜色不敏感 → 更大的区域 |
| `edge_weight` | 2.5 | 边缘作用增强 → 区域严格沿强边缘断开 → 更尊重轮廓。范围 [0.25, 4.0] | 边缘作用减弱 → 区域冒边缘风险合并 → 可能跨线 |

---

### 4d-2. region_scoring — 区域属性评分参数

分割完成后，对每个区域计算以下属性（均为 0~1 之间的得分）:

- **saliency** = 5 种底层特征的加权平均，反映区域视觉突出程度
- **bg_probability** = 5 种背景线索的加权平均，经过 subject 保护衰减
- **subject_confidence** = 3 种主体线索的加权平均

这些属性用于:

1. 生成第 19D `segment_foreground` 特征，直接进入 `weights_add` / `weights_mul` 融合
2. 可选修正 `background_mask_morph` / `background_fg_confidence` 两个传统背景特征
3. 给动态权重提供区域分离度、显著度相关性、前景-背景分离度 3 个指标
4. 计算 `region_color_score`，直接选择最高分区域作为 region color

`segment_foreground = (1 - bg_probability) × 0.65 + subject_confidence × 0.35`，高值表示该像素所在区域更像主体/非背景。它不做 percentile 拉伸，作为 0~1 概率先验直接参与第 19D 权重融合。

#### 显著度权重（region saliency）

| 参数 | 默认 | 调大 → 该特征更主导区域的"显著度"判断 |
|------|------|------|
| `saliency_dct_weight` | 0.20 | DCT 纹理复杂度（平滑区域低，纹理区域高） |
| `saliency_lab_grad_weight` | 0.22 | LAB 梯度强度（结构边缘处高） |
| `saliency_spectral_weight` | 0.28 | 频谱残差（全局视觉弹出度，当前权重最高） |
| `saliency_local_light_weight` | 0.15 | 局部亮度残差（高斯模糊差值） |
| `saliency_local_sat_weight` | 0.15 | 局部饱和度残差（高斯模糊差值） |

#### 背景概率权重（bg_probability）

| 参数 | 默认 | 调大 → 该证据更倾向于把区域判为背景 |
|------|------|------|
| `bg_border_weight` | 0.26 | 区域触碰图像边缘的比例高 → 更像背景 |
| `bg_color_weight` | 0.32 | 区域颜色接近图像边框颜色 → 更像背景（当前最高） |
| `bg_low_saliency_weight` | 0.18 | 区域显著度低 → 更像背景 |
| `bg_low_center_weight` | 0.16 | 区域远离画面中心 → 更像背景 |
| `bg_low_edge_weight` | 0.08 | 区域内部边缘弱 → 更像背景（均匀区域） |

| `subject_protect_strength` | 0.55 | ↑ 主体置信度更强烈地压制背景概率。公式: `protected_bg = bg_raw × (1 − strength × subject_raw × bias)` |

#### 主体置信度权重（subject_confidence）

| 参数 | 默认 | 调大 → 该证据更倾向于把区域判为主体 |
|------|------|------|
| `subject_center_weight` | 0.35 | 区域靠近画面中心 → 更像主体 |
| `subject_saliency_weight` | 0.45 | 区域显著度高 → 更像主体（当前最高权重） |
| `subject_edge_weight` | 0.20 | 区域内部边缘强 → 更像主体（纹理丰富） |

#### 背景修正 & 主色提取

| 参数 | 默认 | 效果 |
|------|------|------|
| `background_correction_strength` | 0.00 | ↑ 区域先验更强势地覆盖原始背景管线结果。0 = 完全用原始背景管线，仅让 segment 通过第 19D 参与融合。对 `background_mask_morph`/`bg_fg_conf` 生效 |
| `border_band` | 8 | ↑ 更多边缘像素参与"边框原型色"采样和 border_ratio 计算，可能混入前景像素。↓ 只用极边缘像素，样本偏少但更纯 |
| `area_power` | 0.55 | ↑ 大面积区域在 color_score 中的权重被指数放大。1.0 → 面积与权重成正比；0.5 → 压缩面积差异 |
| `color_stability_weight` | 0.30 | ↑ 颜色均匀的区域 color_score 更高（压制杂色块）。0 → 只看面积不看均匀度；1 → 均匀度完全支配 |

#### 平铺背景惩罚 & 鲜艳红绿奖励

`region color` 的 Top-1 排名现在额外乘上两个区域色彩因子:

- **背景惩罚**: 大面积 × 低方差/高稳定度 × 低饱和度。用于压制大块灰白、黑白、低饱和平铺背景。
- **红绿奖励**: 大面积 × 高饱和度 × 色相接近红/绿。用于轻微保护大块鲜艳红色/绿色主题色。

| 参数 | 默认 | 效果 |
|------|------|------|
| `bg_flat_penalty_strength` | 0.55 | ↑ 更强烈惩罚大面积、低方差、低饱和区域。0 = 禁用该惩罚 |
| `bg_flat_area_min_ratio` | 0.08 | 区域面积超过该比例后开始触发平铺背景惩罚 |
| `bg_flat_area_full_ratio` | 0.35 | 区域面积超过该比例后平铺背景惩罚满额触发 |
| `bg_flat_sat_threshold` | 0.28 | 饱和度低于该阈值时进入低饱和惩罚，越大越容易惩罚淡色块 |
| `vivid_rg_bonus_strength` | 0.12 | ↑ 更明显奖励大面积鲜艳红/绿色块；建议保持较小，避免红绿背景误伤 |
| `vivid_rg_sat_threshold` | 0.45 | 饱和度高于该阈值后开始触发红绿奖励 |
| `vivid_rg_hue_width` | 38.0 | 红/绿色相容忍宽度，单位度。越大越容易把橙红、黄绿也纳入奖励 |

最终 `region color` 不做多区域加权混合，而是直接采用 `color_score` 最高的单个区域均色。

---

### 4d-3. dynamic_weights — 区域感知动态权重参数

动态权重机制根据每个特征在分割区域上的表现来调整其融合权重:

- **像素统计分** = 方差 / 百分位范围 / 峰度，衡量特征自身的信息量
- **区域区分分** = 该特征是否能在不同区域间产生差异化的值

两部分加权平均后映射为 [min_multiplier, max_multiplier] 的倍增因子，**最终权重 = base_weight × multiplier**。

**区域区分分由 3 个子指标组成:**

- `separation` = between_group_variance / total_variance — 区域间差异 vs 区域内差异
- `saliency_corr` = 特征值与区域显著度的 Pearson 相关系数
- `bg_suppression` = (fg_mean − bg_mean) / 0.45 — 前景区域值是否高于背景区域

| 参数 | 默认 | 效果 |
|------|------|------|
| `enabled` | true | false → region_score 恒为 0，只用像素统计 |
| `pixel_stat_weight` | 0.65 | ↑ 动态权重更取决于特征自身的统计信息量 |
| `region_score_weight` | 0.35 | ↑ 动态权重更取决于特征能否区分不同区域。与 pixel_stat_weight 归一化混合 |
| `separation_weight` | 0.45 | ↑ 在区域评分中更看重"特征值在不同区域间不同"。对有清晰色块边界区分能力的特征更友好 |
| `saliency_corr_weight` | 0.35 | ↑ 更看重特征值与显著度的正相关性。与显著区域一致的 feature 获得更高权重 |
| `bg_suppression_weight` | 0.20 | ↑ 更看重特征在前景/背景间的数值差异。能区分前景（高值）和背景（低值）的特征获更高权重 |

### segment_fusion 典型调参场景

| 场景 | 建议 |
|------|------|
| 主体轮廓清晰、背景单调的图 | 增大 `edge_weight` / `saliency_spectral_weight` |
| 主体与背景颜色相近 | 增大 `color_weight`，降低 `edge_weight` |
| 背景管线误判较多（把主体区域当背景）| 增大 `background_correction_strength` |
| 背景管线漏判较多（背景区域被当成主体）| 减小 `background_correction_strength` |
| 希望 segment 只作为普通特征参与 | 保持 `background_correction_strength: 0`，调 `weights_add/mul.segment_foreground` |
| 分割过碎（太多小区域）| 增大 `min_region_area` / `color_merge_distance` |
| 分割合并过度（大块含异色）| 减小 `color_merge_distance`，增大 `edge_split_strength` |
| 想让区域信息更主导特征权重 | 增大 `region_score_weight`，减小 `pixel_stat_weight` |

---

## 5. 特征权重

两套独立的 19 维权重:

- **`weights_add`**: 加法分支（加权求和）
- **`weights_mul`**: 乘法分支（加权软乘法，各特征概率相乘）

每组权重内部会自动归一化（除以总和），所以不必刻意让总和为 1。
某特征权重设 0 → 该特征不参与融合；设越大 → 该特征对结果影响越大。

### 加法分支特征表

| 特征 | 默认值 | 调大则更关注… |
|------|--------|---------------|
| `dct` | 0.10 | 纯色/平滑背景区域（高频纹理少的地方）被压制 |
| `lab_grad` | 0.02 | 结构边缘（线条、轮廓）被高亮 |
| `spectral` | 0.15 | 全局视觉显著区域（突兀物体）被突出 |
| `global_light` | 0.00 | 偏离整体亮度的色块（亮/暗异常区域） |
| `global_lab_a` | 0.09 | 偏离全局 a\* 红-绿色调的像素（暖调/冷调异常） |
| `global_lab_b` | 0.08 | 偏离全局 b\* 黄-蓝色调的像素 |
| `global_sat` | 0.01 | 偏离全局饱和度的像素（鲜艳/素雅异常） |
| `local_light` | 0.00 | HSL 局部亮度细节（精细光影变化） |
| `local_lab_a` | 0.00 | LAB a\* 局部残差（红-绿轴细节变化） |
| `local_lab_b` | 0.00 | LAB b\* 局部残差（黄-蓝轴细节变化） |
| `local_sat` | 0.01 | HSL 局部饱和度残差（鲜艳色块边缘） |
| `background_mask_morph` | 0.25 | 三阶段硬 mask（1=前景/0=背景）。调大→按背景掩膜严格裁剪 |
| `background_fg_confidence` | 0.65 | 三阶段软置信度（0~1）。调大→更依赖背景管线的主体判断 |
| `subject_prior` | 0.00 | 离画面中心越近权重越高（构图中心偏向） |
| `abs_light` | 0.00 | 绝对 L\* 明度通道（亮→暗直接映射） |
| `abs_lab_a` | 0.035 | 绝对 a\* 红绿色通道（绿→红直接映射） |
| `abs_lab_b` | 0.03 | 绝对 b\* 黄蓝色通道（蓝→黄直接映射） |
| `abs_sat` | 0.06 | 绝对饱和度通道（鲜艳→素雅直接映射） |
| `segment_foreground` | 0.30 | color-segment 派生前景概率。高值 = 区域更像主体/非背景 |

### 乘法分支

乘法分支 = 软乘法: `score = product((ε + (1-ε)*feat_i)^w_i)`

**特性:** 任一特征值接近 0 时，乘积会被"拉低"，所以乘法分支比加法更"保守"。适合用于"所有特征同时认可"的区域才高亮的场景。
当前默认 `weights_mul.segment_foreground = 0.20`，用于让色块前景概率参与保守门控；想让 segment 只做软加分，可把它降到 0。

### 常见调参策略

| 目标 | 建议 |
|------|------|
| 想突出构图主体 | 增大 spectral / lab_grad |
| 想压制纯色背景 | 增大 dct |
| 想保留全局色彩印象 | 增大 global_light + global_lab_a/b |
| 想增强局部纹理细节 | 增大 local_light + local_lab_a/b |
| 想利用背景分割裁剪主体 | 增大 background_mask_morph + background_fg_confidence |
| 想让色块分割参与主体判断 | 增大 `segment_foreground`；若不希望改写传统背景图，保持 `background_correction_strength: 0` |
| 背景管线过度检测时 | 降低 bg_score_threshold / 调大 max_bg_ratio |
| 背景管线检测不足时 | 调高 bg_score_threshold / 增大 border_band |
| 可以某一分支全部设 0 | 等价于只用另一分支 |

---

## 6. Hybrid Fusion 参数

**融合公式:**

```
hybrid = α × add_score + (1-α) × softmul_score
result = hybrid^γ
```

其中 `add_score = Σ(w_i × feat_i)`，`softmul_score = exp(Σ(w_i × ln(ε + (1-ε)×feat_i)))`

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `alpha` | 0.15 | 加法/乘法混合比。0=纯软乘，1=纯加法。偏加法 (α≈1) → 结果更"慷慨"；偏乘法 (α≈0) → 结果更"严格"。建议范围: 0.0~1.0 |
| `gamma` | 1.0 | 最终对比度调整指数。gamma<1 → 整体提亮；gamma=1 → 无变化；gamma>1 → 整体压暗、对比度增强。建议范围: 0.3~2.0 |
| `epsilon` | 0.15 | 软乘法 baseline，防止某特征值为 0 时乘积归零。ε→0 → 乘法效果更强（严格 AND）；ε→1 → 乘法退化≈加法。建议范围: 0.01~0.50 |

---

## 6a. Direct Blend 参数

无阈值过滤的 Hybrid × 原图，以及加权聚类预测色。

**`direct_blend.normalize_before`** — 在无阈值复合图乘原图之前，是否先对 raw `fused_hybrid` 做 [0,1] 归一化。

- `fused_original_hybrid_nothreshold.png` 使用未过滤的 raw `fused_hybrid`
- `weighted cluster` 直接使用过滤后的 `fused_hybrid_filtered` 数值，不再二次归一化，因此会精确响应 `filter.post_normalize_min` / `filter.post_normalize_gamma`
- filtered 聚类的最终簇得分 = `簇大小 × 簇平均权重`，等价于簇内过滤权重总和；不会再用 `weight_sum × count` 把簇大小算两次

| 值 | 效果 |
|----|------|
| false | 直接使用 raw `fused_hybrid` 值 |
| true | 先 min-max 归一化到 [0,1] 再乘原图（适用于 raw hybrid 输出动态范围很窄的场景） |

与 `filter.normalize_before` 独立，可分别配置。

---

## 7. 最终 Fuse 图过滤

两种过滤方案互斥，通过 `filter.method` 选择。若同时配置两种方法（或 method 值不合法），程序启动时会 panic。

### 阈值法 (method = "threshold")

像素亮度 ≥ 阈值时保留，否则置 0。

| 参数 | 说明 |
|------|------|
| `threshold` | 亮度阈值 [0, 1] |
| `normalize_before` | 若为 true，先对亮度做 [0,1] 归一化再比较阈值 |
| `post_normalize` | 若为 true，阈值处理后再对保留的正值像素做归一化，过滤掉的 0 保持不变 |
| `post_normalize_min` | 后处理归一化下限 [0, 1]。0 表示保留区域拉伸到 [0, 1]；0.2 表示拉伸到 [0.2, 1] |
| `post_normalize_gamma` | 后处理 gamma，作用于归一化后的保留区域坐标。<1 放大弱响应，>1 压低弱响应，1 不改变 |

### 分位数法 (method = "quantile")

只保留亮度从高到低前 p% 的像素，其余置黑。

| 参数 | 说明 |
|------|------|
| `quantile` | 保留亮度前百分之几的像素 (0, 100] |
| `post_normalize` | 若为 true，分位数过滤后再对保留的正值像素做归一化 |
| `post_normalize_min` | 后处理归一化下限 [0, 1] |
| `post_normalize_gamma` | 后处理 gamma，必须 > 0 |

无论哪种方法，过滤后保存为 `fused_add_filtered.png` / `fused_softmul_filtered.png` / `fused_hybrid_filtered.png`。

后处理顺序为：

```text
fused_hybrid -> normalize_before(可选, 仅用于阈值比较) -> threshold/quantile 置 0 -> post_normalize(可选) -> gamma
```

`post_normalize` 只重映射过滤后仍大于 0 的像素，因此不会把已剔除的背景重新抬亮。gamma 作用在保留区域的 0~1 归一化坐标上，再映射到 `[post_normalize_min, 1]`，所以不会破坏配置的输出下限。该开关会影响 `Filt Add` / `Filt Mult` / `Filt Hybrid` 三张 filtered 输出，以及使用 `Filt Hybrid` 的 `Orig×Hyb` 和加权聚类输入。

---

## 8. Contact Sheet 拼贴图布局

**仅影响输出排版，不影响特征计算。**

| 参数 | 默认 | 说明 |
|------|------|------|
| `cols` | 10 | 列数 |
| `rows` | 5 | 行数 |
| `pad` | 4 | 单元格间距（像素） |
| `thumb_w` | 240 | 每个单元格的最大宽度（像素），高度按原图比例自动计算 |
| `label_h` | 16 | 单元格底部标签区域高度（像素） |

---

## 9. DCT 纹理复杂度参数

### `dct.high_freq_threshold` (默认 3)

DCT 高频分量判定阈值。对每个 8×8 逐像素滑动窗口做 Type-II DCT，计算高频比率:

```
ratio = Σ_{u+v ≥ T} F(u,v)² / (Σ_{all AC} F(u,v)² + 1e-10)
```

即对角线 u+v=T 以下的系数能量占所有 AC 系数的比例（排除 DC 分量）。

| 阈值 T | AC 系数数量 | 效果 |
|--------|------------|------|
| T=2 | 56/63 | 很多系数算高频 → 纹理检测最敏感 |
| T=3 | 42/63 | 默认值 |
| T=4 | 30/63 | — |
| T=6 | 12/63 | — |
| T=8 | 2/63 | 只有极高频率算纹理 → 平滑区更大 |

| 方向 | 效果 |
|------|------|
| 调大 (如 6~8) | 只有极高频率算纹理 → 平滑区更大，只保留最强纹理 |
| 调小 (如 2~3) | 更多系数算高频 → 纹理检测更敏感，噪声也可能被响应 |

> 注: DCT 块大小固定为 8，窗口以每个像素为中心采样 8×8 邻域（滑动窗口，非分块）。

---

## 10. 频谱残差显著性参数

### `spectral_residual.mean_filter_kernel` (默认 3)

均值滤波核大小，作用于 FFT 对数幅度谱:

```
R(u,v) = ln|F(u,v)| - avg_filter(ln|F(u,v)|, kernel)
```

即"频谱减平均谱"——剩下的就是显著信号的谱残差。

| 方向 | 效果 |
|------|------|
| 核越大 | 平均谱越平滑 → 残差捕获更全局的谱异常（大尺度显著区域） |
| 核越小 | 平均谱更贴近原始 → 残差捕获更精细的谱峰（纹理/边缘） |
| kernel=1 | 均值=原始 → 残差=0 → 所有频率振幅=1 → 退化为相位谱显著图（POT/边缘图） |

实际核尺寸 = kernel（奇数时）或 kernel+1（偶数时）。建议范围: 1~15。

### `spectral_residual.gaussian_sigma` (默认 3.0)

IFFT 重构后的 saliency map Gaussian blur 标准差。用于平滑重构噪声，使显著区域更连续。

| 方向 | 效果 |
|------|------|
| sigma 越大 | 平滑更强 → 显著区域更模糊、更大片 |
| sigma 越小 | 保留更多高频细节 → 显著图更精细 |
| 0 | 不 blur |

建议范围: 0~10。

### `spectral_residual.gamma` (默认 1.0)

FFT 前对 L\* 通道（亮度，始终 [0,1]）做 gamma 校正。a\*、b\* 不做处理。

| 值 | 效果 |
|----|------|
| gamma<1 | 压缩，提亮暗部细节，使低频区域的谱残差更显著 |
| gamma>1 | 扩展，压制暗部 |
| =1.0 | 无变化 |

建议范围: 0.3~3.0。

### `spectral_residual.post_gamma` (默认 1.0)

后处理 gamma 压缩（L₂ 融合 + 归一化后对最终显著图做 powf）。

公式: `S_final = S^post_gamma`（作用于已归一化到 [0,1] 的融合显著图）

| 值 | 效果 |
|----|------|
| <1 (如 0.5) | 凹函数，抬升低值区域 → 更多像素获得显著响应（更"慷慨"） |
| >1 (如 2.0) | 凸函数，压制低值区域 → 只有最高显著峰幸存（更"严格"、更稀疏） |
| =1.0 | 跳过 powf，无变化 |

建议范围: 0.3~3.0。

---

## 11. 印象色聚类参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `k` | 4 | 聚类数量。调大→更细分，调小→更概括 |
| `max_iter` | 10 | K-means Lloyd 迭代次数上限。通常 5~10 次即可收敛 |
| `sample_method` | stride | "stride" — 间隔网格采样；"all" — 全部 filtered > 0 像素 |
| `sample_stride` | 4 | 网格采样步长。stride=1 等价于 "all"。stride=4 → 每 4 像素取一个，样本量约 1/16。建议范围: 2~8 |
| `seed` | 42 | K-Means++ 初始化的随机数种子。固定种子保证同图同参数结果可复现 |

---

## 12. 动态特征权重（Dynamic Feature Weights）

根据每张特征图自身的统计量动态调整融合权重。

**核心思想:** 空间变化强的特征自动提高权重，接近常量的特征自动降低权重。

**公式:** `dynamic_weight = base_weight × multiplier`（base_weight == 0 时，dynamic_weight 仍为 0）

### 顶层参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `enabled` | true | 是否启用动态权重。false 时行为与旧版完全一致 |
| `min_multiplier` | 0.35 | multiplier 下限 — 信息量低的特征最多降到原权重的 min_multiplier 倍 |
| `max_multiplier` | 1.85 | multiplier 上限 — 信息量高的特征最多升到原权重的 max_multiplier 倍 |
| `eps` | 1e-6 | 防止除零的小常数（峰度计算的 mean 分母） |

### 统计量参考值

三项统计量的参考值（经验值），得分 = clamp(stat / ref, 0, 1):

| 参数 | 默认 | 说明 |
|------|------|------|
| `variance_ref` | 0.05 | 方差参考值 |
| `range_ref` | 0.65 | 百分位范围参考值 |
| `peakiness_ref` | 4.0 | 峰度参考值 |

### stat_mix — 三项统计量混合权重

| 参数 | 默认 |
|------|------|
| `variance` | 0.55 |
| `range` | 0.25 |
| `peakiness` | 0.20 |

### percentile — 方差计算前的 clip 范围

| 参数 | 默认 | 说明 |
|------|------|------|
| `low` | 2.0 | 剔除极端离群值下界 |
| `high` | 98.0 | 剔除极端离群值上界 |

### per_feature — 各特征允许动态权重

| 特征 | 默认 | 说明 |
|------|------|------|
| `dct` | enabled | 允许动态 |
| `lab_grad` | enabled | 允许动态 |
| `spectral` | enabled | 允许动态 |
| `global_light` ~ `global_sat` | enabled | 全局残差系列允许动态 |
| `local_light` ~ `local_sat` | enabled | 局部残差系列允许动态 |
| `abs_light` ~ `abs_sat` | enabled | 绝对通道系列允许动态 |
| `background_mask_morph` | disabled | 结构性先验，不应随图像统计量大幅变化 |
| `background_fg_confidence` | disabled | 结构性先验 |
| `subject_prior` | disabled | 结构性先验 |
| `segment_foreground` | enabled | 区域前景先验，允许按区域分离度/背景抑制能力调整 multiplier |
