# 🎨 Color Extract — 图片主色/调色盘提取工具

基于 Rust 的高性能图片颜色提取 demo，支持 **4 种算法 × 5 个色彩空间**，输出可视化 HTML 报告。

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
# 图片放在 imgs/ 目录下（支持 jpg/jpeg/png）
cargo run --release

# 自定义参数
cargo run --release -- <max_dim> <output_path>
# 例如：
cargo run --release -- 512 output/my-results.html
```

输出文件默认位于 `output/results.html`，用浏览器打开即可。

## 项目结构

```
color-extract/
├── Cargo.toml
├── src/
│   ├── main.rs                  # CLI 入口，并行调度
│   ├── img.rs                   # 图片加载、缩放、Base64 缩略图
│   ├── colorspace.rs            # 5 个色彩空间的转换 + CAM16-UCS 反算
│   ├── html.rs                  # HTML 报告生成（暗色主题）
│   ├── timing.rs                # 耗时测量工具
│   └── algorithms/
│       ├── mod.rs               # Algorithm trait + PaletteEntry 类型
│       ├── kmeans.rs            # KMeans++ (linfa-clustering)
│       ├── minibatch_kmeans.rs  # Mini-Batch KMeans (batch=2048)
│       ├── median_cut.rs        # Median Cut 自定义实现
│       └── octree.rs            # Octree 八叉树量化（双策略）
├── imgs/                        # 输入图片目录
└── output/                      # 生成的 HTML 报告
```

## 性能

以 3 张动漫插画（~800×1200，缩放至 ≤1024）在 Ryzen 处理器上测试：

| 指标 | 数值 |
|------|------|
| 总组合数 | 60 (3图 × 4算法 × 5空间) |
| 总耗时 | ~5 秒 (release) |
| 最快算法 | Median Cut (~170ms/组合) |
| 最慢算法 | KMeans++ (~2-4s/组合) |

## 依赖

| Crate | 用途 |
|-------|------|
| `image` 0.25 | 图片加载/缩放 |
| `palette` 0.7 | 全部 5 个色彩空间 + CAM16 |
| `linfa-clustering` 0.8 | KMeans++ / Mini-Batch KMeans |
| `ndarray` 0.15 | 矩阵计算 |
| `rayon` 1 | 多线程并行 |
| `base64` 0.22 | 图片内嵌 HTML |

## 构建

```bash
# 需要 Rust 1.94+
cargo build --release
```

## 技术细节

### CAM16 色彩外观模型

通过 `palette` crate 实现，使用 sRGB 标准观看条件（40 cd/m²）。CAM16-UCS J'a'b' 坐标的反向转换使用 Li et al. (2017) 的数学公式手动实现，绕过了 palette 的类型系统限制。

### 八叉树双策略

- **RGB 空间**：经典位索引八叉树，直接对 0-255 整数坐标按位分组
- **其他空间**：范围索引八叉树，在每个节点计算坐标范围的二分中点来分组

### 主色提取

主色来自算法内部状态，而非调色盘排序结果：
- KMeans/Mini-Batch：像素数最多的聚类的质心
- Median Cut：最大 bucket 的均值
- Octree：像素数最多的叶子节点的均值

## License

MIT
