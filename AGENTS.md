# Color Extract — AI Agent Guide

> Rust workspace (edition 2024) — 8 crates for image color extraction, feature map visualization, and saliency detection.

## Build & Run

```bash
# Build everything (release)
cargo build --release

# Run specific crate
cargo run --release -p feature-fuse          # 全链路特征图 + Hybrid Fusion
cargo run --release -p dct-extract            # DCT 增强颜色提取
cargo run --release -p dct-viz                # DCT 纹理复杂度可视化
cargo run --release -p lab-gradient           # LAB Sobel 梯度
cargo run --release -p spectral-residual      # 频谱残差显著性
cargo run --release -p global-residual        # 全局均值残差
cargo run --release -p residual-viz           # 高斯残差
cargo run -p img-compare                      # Web 对比服务 (debug)

# Run root crate (经典调色盘提取)
cargo run --release

# Arguments pattern: [max_dim] [output_path]
cargo run --release -p dct-extract -- 512 output/my-report.html
```

## Architecture

| Crate | Purpose |
|-------|---------|
| `color-extract` (root) | 4 algorithms × 5 color spaces palette extraction |
| `dct-extract` | DCT-enhanced color extraction with scoring system |
| `feature-fuse` | 7 feature maps → percentile normalize → hybrid fusion |
| `dct-viz` | 8×8 block DCT heatmap visualization |
| `lab-gradient` | LAB Sobel gradient → RGB composite |
| `spectral-residual` | 2D FFT spectral residual saliency |
| `global-residual` | HSL/LAB global mean residual |
| `residual-viz` | Gaussian blur residual (HSL + LAB) |
| `img-compare` | Web-based layer compositing comparison |

## Key Conventions

- **Error handling**: `anyhow::Result` / `anyhow::Context` throughout.
- **Image loading**: `image` 0.25, Lanczos3 resize, longest side ≤ max_dim (default 1024).
- **Color spaces**: `palette` 0.7 crate for CIELAB, Oklab, HSL, CAM16-UCS.
- **Parallelism**: `rayon` for per-pixel or per-image parallel work.
- **CLI args**: `env::args().nth(1)` = max_dim, `nth(2)` = output path.
- **Input**: `imgs/` dir (jpg/jpeg/png). **Output**: `output/` dir.
- **Code style**: Section banners `// =====`, Chinese doc comments for complex logic.
- **Tests**: None currently — verify by running crates on `imgs/`.

## feature-fuse Specifics

- Config in `feature-fuse/params.yaml` — edit then run, no recompile needed.
- See [README.md](README.md#feature-fuse--全链路特征图计算--hybrid-fusion) for 7 feature types and fusion parameters.
- 12 output PNGs per image (7 features + 3 fused + contact sheet + resized original).

## dct-extract Specifics

- Scoring parameters at top of `dct-extract/src/cluster_4d.rs` — edit & recompile.
- See [README.md](README.md#dct-extract--dct-增强颜色提取) for scoring formula.
- BFS boundary connectivity for background detection.

## Common Pitfalls

1. **edition 2024** — requires Rust 1.84+. Use `cargo --version` to check.
2. **`imgs/` must exist** — all crates read from this directory; create it if missing.
3. **feature-fuse filter params** — `threshold` and `quantile` are mutually exclusive (validated at runtime).
4. **img-compare**: runs in debug mode (actix-web), no `--release` needed.
5. **font8x8 bit order**: `BASIC_LEGACY` uses MSB-left — `(row_data >> col) & 1`.

## Documentation

- [README.md](README.md) — full project documentation for all crates.
- `output/` — generated HTML reports and visualizations.
