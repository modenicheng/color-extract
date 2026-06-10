// =============================================================================
// Region type — connected component statistics
// =============================================================================

/// A single connected region's statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// Final contiguous region ID (0, 1, ...)
    pub id: usize,
    /// Cluster index this region belongs to (0..N-1)
    pub cluster_id: usize,
    /// Region area in pixels
    pub area: usize,
    /// Centroid coordinates (x, y), sub-pixel precision
    pub centroid: (f64, f64),
    /// Bounding box (min_x, min_y, max_x, max_y), inclusive
    pub bbox: (u32, u32, u32, u32),
    /// Pixel count (same as area, kept for downstream compatibility)
    pub pixel_count: usize,
}
