// =============================================================================
// YAML 参数定义 + 加载 — 颜色分割参数
// =============================================================================
//
// 本模块定义了 `SegmentParams` 配置结构体，支持从 YAML 文件加载分割参数。
// 所有字段均有独立的默认值函数，通过 `#[serde(default = "...")]` 显式指定。

use serde::{Deserialize, Serialize};

// =============================================================================
// ColorSpace 枚举
// =============================================================================

/// 分割运算的色彩空间
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// CIELAB L\*a\*b\* (感知均匀)
    #[serde(rename = "Lab")]
    Lab,
    /// Oklab (现代感知均匀)
    #[serde(rename = "Oklab")]
    Oklab,
    /// sRGB
    #[serde(rename = "Rgb")]
    Rgb,
    /// HSL (色相-饱和度-明度)
    #[serde(rename = "Hsl")]
    Hsl,
}

impl Default for ColorSpace {
    fn default() -> Self {
        Self::Lab
    }
}

// =============================================================================
// SegmentParams 主结构体
// =============================================================================

/// 颜色分割算法的全部可调参数。
///
/// 所有字段均有合理的默认值，可通过 YAML 文件按需覆盖。
/// 加载方式：`serde_yaml::from_str::<SegmentParams>(&yaml)`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentParams {
    /// 最大聚类数 — 控制分割粒度上限
    #[serde(default = "default_max_clusters")]
    pub max_clusters: usize,

    /// 四叉树最大递归深度
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,

    /// 方差阈值 — 低于此值停止切分
    #[serde(default = "default_variance_threshold")]
    pub variance_threshold: f64,

    /// 最小聚类面积比 — 低于此比例的聚类被标记为小区域
    #[serde(default = "default_min_cluster_area_ratio")]
    pub min_cluster_area_ratio: f64,

    /// 最小区域像素数 — 小于此值的独立区域被合并
    #[serde(default = "default_min_region_area")]
    pub min_region_area: usize,

    /// 边缘强度阈值 — 边缘检测的灵敏度
    #[serde(default = "default_edge_threshold")]
    pub edge_threshold: f64,

    /// 边缘引导分裂强度
    #[serde(default = "default_edge_split_strength")]
    pub edge_split_strength: f64,

    /// 边缘引导合并强度
    #[serde(default = "default_edge_merge_strength")]
    pub edge_merge_strength: f64,

    /// 分割使用的色彩空间
    #[serde(default = "default_color_space")]
    pub color_space: ColorSpace,

    /// 是否合并小区域
    #[serde(default = "default_merge_small_regions")]
    pub merge_small_regions: bool,

    /// 是否平滑区域边界
    #[serde(default = "default_smooth_boundaries")]
    pub smooth_boundaries: bool,

    /// 边界平滑半径（像素）
    #[serde(default = "default_smooth_radius")]
    pub smooth_radius: u32,
}

// =============================================================================
// 默认值辅助函数
// =============================================================================

fn default_max_clusters() -> usize {
    32
}

fn default_max_depth() -> usize {
    8
}

fn default_variance_threshold() -> f64 {
    1.0
}

fn default_min_cluster_area_ratio() -> f64 {
    0.001
}

fn default_min_region_area() -> usize {
    50
}

fn default_edge_threshold() -> f64 {
    0.15
}

fn default_edge_split_strength() -> f64 {
    0.4
}

fn default_edge_merge_strength() -> f64 {
    0.05
}

fn default_color_space() -> ColorSpace {
    ColorSpace::Lab
}

fn default_merge_small_regions() -> bool {
    true
}

fn default_smooth_boundaries() -> bool {
    true
}

fn default_smooth_radius() -> u32 {
    1
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            max_clusters: default_max_clusters(),
            max_depth: default_max_depth(),
            variance_threshold: default_variance_threshold(),
            min_cluster_area_ratio: default_min_cluster_area_ratio(),
            min_region_area: default_min_region_area(),
            edge_threshold: default_edge_threshold(),
            edge_split_strength: default_edge_split_strength(),
            edge_merge_strength: default_edge_merge_strength(),
            color_space: default_color_space(),
            merge_small_regions: default_merge_small_regions(),
            smooth_boundaries: default_smooth_boundaries(),
            smooth_radius: default_smooth_radius(),
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

        assert_eq!(p.max_clusters, 32);
        assert_eq!(p.max_depth, 8);
        assert_eq!(p.variance_threshold, 1.0);
        assert_eq!(p.min_cluster_area_ratio, 0.001);
        assert_eq!(p.min_region_area, 50);
        assert_eq!(p.edge_threshold, 0.15);
        assert_eq!(p.edge_split_strength, 0.4);
        assert_eq!(p.edge_merge_strength, 0.05);
        assert_eq!(p.color_space, ColorSpace::Lab);
        assert!(p.merge_small_regions);
        assert!(p.smooth_boundaries);
        assert_eq!(p.smooth_radius, 1);
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
        let yaml = "max_clusters: 16\ncolor_space: Oklab\n";
        let p: SegmentParams = serde_yaml::from_str(yaml).expect("parse partial YAML");

        // 覆盖的字段
        assert_eq!(p.max_clusters, 16);
        assert_eq!(p.color_space, ColorSpace::Oklab);

        // 其余字段保持默认
        assert_eq!(p.max_depth, 8);
        assert_eq!(p.variance_threshold, 1.0);
        assert_eq!(p.min_cluster_area_ratio, 0.001);
        assert_eq!(p.min_region_area, 50);
        assert_eq!(p.edge_threshold, 0.15);
        assert_eq!(p.edge_split_strength, 0.4);
        assert_eq!(p.edge_merge_strength, 0.05);
        assert!(p.merge_small_regions);
        assert!(p.smooth_boundaries);
        assert_eq!(p.smooth_radius, 1);
    }

    /// 空 YAML 应产生全默认值
    #[test]
    fn test_empty_yaml() {
        let yaml = "{}";
        let p: SegmentParams = serde_yaml::from_str(yaml).expect("parse empty YAML");
        assert_eq!(p, SegmentParams::default());
    }
}
