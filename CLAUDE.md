# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Requires Rust 1.84+ (edition 2024)
cargo build --release

# Build a specific crate
cargo build --release -p <crate>

# Run a crate
cargo run --release -p <crate>
cargo run --release -p <crate> -- [max_dim] [output_path]

# Run tests (workspace-wide)
cargo test --release

# Run tests for a specific crate
cargo test --release -p <crate>
```

## Workspace Overview

This is a Rust workspace of 10 crates for image color analysis, palette extraction, and visualization. All crates share the pattern: load images from `imgs/` → Lanczos3 scale to `max_dim` → process in parallel via Rayon → output to `output/`.

### Crate Map

| Crate | Purpose |
|-------|---------|
| `color-extract` (root) | Classic palette extraction: 4 algorithms × 5 color spaces, self-contained HTML report |
| `dct-extract` | DCT-enhanced color extraction with 7-factor scoring system and BFS background detection |
| `dct-viz` | DCT texture complexity heatmap/gray/overlay visualization |
| `spectral-residual` | 2D FFT spectral residual saliency detection (Hou & Zhang 2007) |
| `global-residual` | Per-pixel deviation from global HSL/LAB means |
| `residual-viz` | Gaussian blur residual — |original − blurred| across HSL/LAB channels |
| `lab-gradient` | 3-channel LAB Sobel gradient mapped to a single RGB image |
| `img-compare` | actix-web server for interactive layer-compositing comparison |
| `color-segment` | **Active development.** Color-based region segmentation for anime-style images |
| `feature-fuse` | 18-feature full pipeline: compute → normalize → hybrid fusion → weighted k-means impression color |
| `impression-color-extract` | Background/foreground separation for impression color extraction |

### Key Shared Dependencies

- **Color science:** `palette` 0.7 (CIELAB, Oklab, HSL, CAM16), `image` 0.25
- **Clustering:** `linfa-clustering` 0.8 (KMeans++, Mini-Batch KMeans)
- **FFT:** `rustfft` 6 (spectral residual crates)
- **Parallelism:** `rayon` 1 (all crates)
- **Config:** `serde_yaml` (newer crates use `params.yaml` for runtime tuning without recompilation)

## Architecture Notes

### color-segment (active development — uncommitted `anime.rs`)

This crate has two segmentation code paths:

1. **Legacy 4-phase pipeline** (older modules): `quantize.rs` (Median Cut in CIELAB) → `region.rs` (4-connected CCL) → `edge.rs` (LAB Sobel smoothstep) → `refine.rs` (edge-aware merge + split)

2. **New fast path** (`anime.rs`, uncommitted): Single-scan Union-Find region growing on YCoCg + LAB features. The `segment()` entry in `lib.rs` dispatches directly to `anime::segment_anime_blocks()`. Key characteristics:
   - Extracts RGB/YCoCg/LAB features once
   - LAB Sobel edges with gamma compression for edge walls
   - Scan-order UF region growing using color distance + edge walls as boundaries
   - Adjacency-graph merge with small-region absorption via scoring heuristics
   - Parameters come from `params.yaml` via `SegmentParams`
   - Set `COLOR_SEGMENT_PROFILE=1` env var for timing breakdown

The older modules (`edge.rs`, `quantize.rs`, `refine.rs`, `region.rs`) still exist but are not in the active call path through `segment()`.

### Feature Fusion Pipeline (feature-fuse)

Computes 18 normalized feature maps per image: DCT complexity, LAB gradient, spectral residual, global residuals (light, a*, b*, sat), local Gaussian residuals (light, a*, b*, sat), background features (mask + fg confidence), subject prior, and absolute channels (L*, a*, b*, sat). These are fused via additive + soft-multiplicative + hybrid (α-blend of both) into a saliency heatmap. Weighted k-means on the hybrid map extracts an "impression color."

### Background Detection (shared pattern)

Multiple crates implement BFS-based background detection: starting from image borders, expanding through similar-colored pixels, computing connectivity scores. The `dct-extract` and `feature-fuse` crates have independent implementations of this pattern.

### YAML-based Parameter Tuning

The crates `color-segment`, `feature-fuse`, and `impression-color-extract` load parameters from their respective `params.yaml` files. These are deserialized via `serde_yaml` into typed structs with `#[serde(default)]` — all fields have defaults, so the YAML only needs to specify overrides. CLI args can override `max_dim` and output path.

### Output Pattern

- Classic crates: single self-contained HTML with Base64-embedded images (dark theme)
- Newer crates: per-image output directories + overview HTML + contact sheet PNG + individual feature PNGs
- All visualizations go under `output/<crate>/`

## Git Conventions

- Commit messages follow `type(crate): description` format (e.g., `feat(color-segment): improve edge detection`)
- Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
