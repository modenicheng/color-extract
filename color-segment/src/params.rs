// =============================================================================
// YAML 参数定义 + 加载 — 颜色分割参数
// =============================================================================
//
// 本模块定义了 `SegmentParams` 配置结构体，支持从 YAML 文件加载分割参数。
// 所有字段均有独立的默认值函数，通过 `#[serde(default = "...")]` 显式指定。

use serde::{Deserialize, Serialize};

// =============================================================================
// SegmentParams 主结构体
// =============================================================================

/// 颜色分割算法的全部可调参数。
///
/// 所有字段均有合理的默认值，可通过 YAML 文件按需覆盖。
/// 加载方式：`serde_yaml::from_str::<SegmentParams>(&yaml)`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentParams {
    /// 预处理最大边长 — 输入图像缩放后宽高均不超过该值；小图保持原尺寸
    #[serde(default = "default_preprocess_max_dim")]
    pub preprocess_max_dim: u32,

    /// 最小区域像素数 — 小于此值的独立区域被合并
    #[serde(default = "default_min_region_area")]
    pub min_region_area: usize,

    /// 最小聚类面积比 — 低于此比例的聚类被标记为小区域
    #[serde(default = "default_min_cluster_area_ratio")]
    pub min_cluster_area_ratio: f64,

    /// 边缘强度阈值 — 边缘检测的灵敏度
    #[serde(default = "default_edge_threshold")]
    pub edge_threshold: f64,

    /// 边缘引导分裂强度 — 控制区域生长时的边缘壁垒和合并守卫
    #[serde(default = "default_edge_split_strength")]
    pub edge_split_strength: f64,

    /// 边缘 gamma 压缩。小于 1 会抬高中等边缘，使线稿/色块边界更突出
    #[serde(default = "default_edge_gamma")]
    pub edge_gamma: f64,

    /// 边缘引导合并强度
    #[serde(default = "default_edge_merge_strength")]
    pub edge_merge_strength: f64,

    /// 相邻区域颜色合并阈值（RGB 距离，越大越容易合并相近颜色）
    #[serde(default = "default_color_merge_distance")]
    pub color_merge_distance: f64,

    /// 小区域吸收到邻接区域时允许的最大颜色距离
    #[serde(default = "default_small_region_color_distance")]
    pub small_region_color_distance: f64,

    /// 是否合并小区域
    #[serde(default = "default_merge_small_regions")]
    pub merge_small_regions: bool,

    /// 形态学开运算半径 (0 = 禁用)。腐蚀再膨胀，清除边缘图中的孤立噪声
    #[serde(default = "default_morph_open_radius")]
    pub morph_open_radius: u8,

    /// 形态学闭运算半径 (0 = 禁用)。膨胀再腐蚀，弥合边缘图中的断裂缺口
    #[serde(default = "default_morph_close_radius")]
    pub morph_close_radius: u8,
}

// =============================================================================
// 默认值辅助函数
// =============================================================================

fn default_preprocess_max_dim() -> u32 {
    512
}

fn default_min_region_area() -> usize {
    50
}

fn default_min_cluster_area_ratio() -> f64 {
    0.001
}

fn default_edge_threshold() -> f64 {
    0.15
}

fn default_edge_split_strength() -> f64 {
    0.4
}

fn default_edge_gamma() -> f64 {
    0.65
}

fn default_edge_merge_strength() -> f64 {
    0.05
}

fn default_color_merge_distance() -> f64 {
    8.0
}

fn default_small_region_color_distance() -> f64 {
    24.0
}

fn default_merge_small_regions() -> bool {
    true
}

fn default_morph_open_radius() -> u8 {
    0
}

fn default_morph_close_radius() -> u8 {
    0
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            preprocess_max_dim: default_preprocess_max_dim(),
            min_region_area: default_min_region_area(),
            min_cluster_area_ratio: default_min_cluster_area_ratio(),
            edge_threshold: default_edge_threshold(),
            edge_split_strength: default_edge_split_strength(),
            edge_gamma: default_edge_gamma(),
            edge_merge_strength: default_edge_merge_strength(),
            color_merge_distance: default_color_merge_distance(),
            small_region_color_distance: default_small_region_color_distance(),
            merge_small_regions: default_merge_small_regions(),
            morph_open_radius: default_morph_open_radius(),
            morph_close_radius: default_morph_close_radius(),
        }
    }
}

// =============================================================================
// 单元测试
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 `Default` 为每个字段返回正确的默认值
    #[test]
    fn test_default_values() {
        let p = SegmentParams::default();

        assert_eq!(p.preprocess_max_dim, 512);
        assert_eq!(p.min_region_area, 50);
        assert_eq!(p.min_cluster_area_ratio, 0.001);
        assert_eq!(p.edge_threshold, 0.15);
        assert_eq!(p.edge_split_strength, 0.4);
        assert_eq!(p.edge_gamma, 0.65);
        assert_eq!(p.edge_merge_strength, 0.05);
        assert_eq!(p.color_merge_distance, 8.0);
        assert_eq!(p.small_region_color_distance, 24.0);
        assert!(p.merge_small_regions);
    }

    /// YAML 完整序列化-反序列化 round-trip：序列化默认值 → 反序列化 → 应与原始相等
    #[test]
    fn test_yaml_round_trip() {
        let original = SegmentParams::default();
        let yaml = serde_yaml::to_string(&original).expect("serialize");
        let restored: SegmentParams = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(original, restored);
    }

    /// YAML 部分覆盖：仅指定部分字段，其余应保持默认
    #[test]
    fn test_partial_yaml_override() {
        let yaml = "preprocess_max_dim: 768\nedge_threshold: 0.08\ncolor_merge_distance: 5.0\n";
        let p: SegmentParams = serde_yaml::from_str(yaml).expect("parse partial YAML");

        // 覆盖的字段
        assert_eq!(p.preprocess_max_dim, 768);
        assert_eq!(p.edge_threshold, 0.08);
        assert_eq!(p.color_merge_distance, 5.0);

        // 其余字段保持默认
        assert_eq!(p.min_region_area, 50);
        assert_eq!(p.min_cluster_area_ratio, 0.001);
        assert_eq!(p.edge_split_strength, 0.4);
        assert_eq!(p.edge_gamma, 0.65);
        assert_eq!(p.edge_merge_strength, 0.05);
        assert_eq!(p.small_region_color_distance, 24.0);
        assert!(p.merge_small_regions);
    }

    /// 空 YAML 应产生全默认值
    #[test]
    fn test_empty_yaml() {
        let yaml = "{}";
        let p: SegmentParams = serde_yaml::from_str(yaml).expect("parse empty YAML");
        assert_eq!(p, SegmentParams::default());
    }
}
