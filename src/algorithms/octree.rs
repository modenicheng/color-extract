use super::{AlgorithmResult, PaletteEntry, make_entry, sort_by_lightness};
use crate::colorspace::ColorSpace;
use crate::timing::timed;
use anyhow::Result;
use std::array;

// ---------------------------------------------------------------------------
// Constants & data structures
// ---------------------------------------------------------------------------

const MAX_DEPTH: usize = 8;

struct OctreeNode {
    children: [Option<usize>; 8], // indices into arena
    pixel_count: u64,
    sum_r: f64,
    sum_g: f64,
    sum_b: f64, // accumulated coords for mean
    is_leaf: bool,
}

struct OctreeQuantizer {
    nodes: Vec<OctreeNode>, // arena
    root: usize,
    leaf_count: usize,
    reducible: [Vec<usize>; MAX_DEPTH + 1], // reducible nodes per level
}

// ---------------------------------------------------------------------------
// Octant-index helpers
// ---------------------------------------------------------------------------

/// Bit-based octant index (for RGB in 0..255 integer space).
fn octant_index_bit(rgb: [f64; 3], depth: usize) -> usize {
    let r = rgb[0].round() as u8;
    let g = rgb[1].round() as u8;
    let b = rgb[2].round() as u8;
    let shift = 7 - depth;
    let ri = ((r >> shift) & 1) as usize;
    let gi = ((g >> shift) & 1) as usize;
    let bi = ((b >> shift) & 1) as usize;
    (ri << 2) | (gi << 1) | bi
}

/// Range-based octant index (for CIELAB, Oklab, HSL, CAM16).
fn octant_index_range(coord: [f64; 3], min: [f64; 3], max: [f64; 3]) -> usize {
    let mid = [
        (min[0] + max[0]) / 2.0,
        (min[1] + max[1]) / 2.0,
        (min[2] + max[2]) / 2.0,
    ];
    let mut idx = 0;
    if coord[0] > mid[0] {
        idx |= 1;
    }
    if coord[1] > mid[1] {
        idx |= 2;
    }
    if coord[2] > mid[2] {
        idx |= 4;
    }
    idx
}

// ---------------------------------------------------------------------------
// OctreeQuantizer implementation
// ---------------------------------------------------------------------------

impl OctreeQuantizer {
    fn new() -> Self {
        let mut nodes = Vec::new();
        let root = nodes.len();
        nodes.push(OctreeNode {
            children: [None; 8],
            pixel_count: 0,
            sum_r: 0.0,
            sum_g: 0.0,
            sum_b: 0.0,
            is_leaf: true,
        });
        Self {
            nodes,
            root,
            leaf_count: 0, // root is empty container, not a data leaf
            reducible: array::from_fn(|_| Vec::new()),
        }
    }

    // --- Bit-based insertion (RGB) ---

    fn insert_bit(&mut self, node_idx: usize, coords: [f64; 3], depth: usize) {
        if depth == MAX_DEPTH {
            let node = &mut self.nodes[node_idx];
            node.pixel_count += 1;
            node.sum_r += coords[0];
            node.sum_g += coords[1];
            node.sum_b += coords[2];
            return;
        }

        let octant = octant_index_bit(coords, depth);
        let child_opt = self.nodes[node_idx].children[octant];

        let child_idx = match child_opt {
            Some(idx) => idx,
            None => {
                let child_idx = self.nodes.len();
                self.nodes.push(OctreeNode {
                    children: [None; 8],
                    pixel_count: 0,
                    sum_r: 0.0,
                    sum_g: 0.0,
                    sum_b: 0.0,
                    is_leaf: true,
                });
                self.nodes[node_idx].children[octant] = Some(child_idx);
                self.leaf_count += 1;

                // First child: parent transitions leaf → internal
                if self.nodes[node_idx].is_leaf {
                    self.nodes[node_idx].is_leaf = false;
                    self.leaf_count -= 1; // parent no longer a leaf
                    self.reducible[depth].push(node_idx);
                }

                child_idx
            }
        };

        self.insert_bit(child_idx, coords, depth + 1);
    }

    // --- Range-based insertion (non-RGB) ---

    fn insert_range(
        &mut self,
        node_idx: usize,
        coords: [f64; 3],
        depth: usize,
        min: [f64; 3],
        max: [f64; 3],
    ) {
        if depth == MAX_DEPTH {
            let node = &mut self.nodes[node_idx];
            node.pixel_count += 1;
            node.sum_r += coords[0];
            node.sum_g += coords[1];
            node.sum_b += coords[2];
            return;
        }

        let octant = octant_index_range(coords, min, max);
        let child_opt = self.nodes[node_idx].children[octant];

        let child_idx = match child_opt {
            Some(idx) => idx,
            None => {
                let child_idx = self.nodes.len();
                self.nodes.push(OctreeNode {
                    children: [None; 8],
                    pixel_count: 0,
                    sum_r: 0.0,
                    sum_g: 0.0,
                    sum_b: 0.0,
                    is_leaf: true,
                });
                self.nodes[node_idx].children[octant] = Some(child_idx);
                self.leaf_count += 1;

                if self.nodes[node_idx].is_leaf {
                    self.nodes[node_idx].is_leaf = false;
                    self.leaf_count -= 1; // parent no longer a leaf
                    self.reducible[depth].push(node_idx);
                }

                child_idx
            }
        };

        // Compute child's bounding box
        let mid = [
            (min[0] + max[0]) / 2.0,
            (min[1] + max[1]) / 2.0,
            (min[2] + max[2]) / 2.0,
        ];
        let mut child_min = min;
        let mut child_max = max;
        if octant & 1 != 0 {
            child_min[0] = mid[0];
        } else {
            child_max[0] = mid[0];
        }
        if octant & 2 != 0 {
            child_min[1] = mid[1];
        } else {
            child_max[1] = mid[1];
        }
        if octant & 4 != 0 {
            child_min[2] = mid[2];
        } else {
            child_max[2] = mid[2];
        }

        self.insert_range(child_idx, coords, depth + 1, child_min, child_max);
    }

    // --- Reduction: merge leaves until leaf_count ≤ target_k ---

    fn reduce(&mut self, target_k: usize) {
        while self.leaf_count > target_k {
            // Find deepest level with a non-empty reducible list
            let deepest = (0..=MAX_DEPTH)
                .rev()
                .find(|&d| !self.reducible[d].is_empty());
            let level = match deepest {
                Some(l) => l,
                None => break, // nothing left to reduce
            };

            let node_idx = self.reducible[level].pop().unwrap();

            // Collect current leaf-child indices
            let child_indices: Vec<usize> = self.nodes[node_idx]
                .children
                .iter()
                .filter_map(|&c| c)
                .collect();
            let nc = child_indices.len();
            if nc == 0 {
                continue; // already fully reduced
            }

            // leaf_count after a full merge: lose nc children, gain this node as leaf
            let after_full = self.leaf_count.saturating_sub(nc).saturating_add(1);

            if after_full >= target_k {
                // Full merge — safe, won't undershoot target
                self.merge_children_into(node_idx, &child_indices, true);
                self.leaf_count = after_full;
            } else {
                // Partial merge — merge only the smallest children to hit target_k exactly
                let need = self.leaf_count.saturating_sub(target_k); // how many children to merge

                if need < nc {
                    // Sort children by pixel_count (ascending — smallest first)
                    let mut sorted: Vec<(usize, u64)> = child_indices
                        .iter()
                        .map(|&idx| (idx, self.nodes[idx].pixel_count))
                        .collect();
                    sorted.sort_by_key(|&(_, c)| c);

                    let to_merge: Vec<usize> =
                        sorted.iter().take(need).map(|&(idx, _)| idx).collect();

                    // Accumulate merged children into the parent
                    self.merge_children_into(node_idx, &to_merge, false);
                    self.leaf_count -= need;

                    // Remove merged children from parent's children array
                    for cidx in &to_merge {
                        for slot in &mut self.nodes[node_idx].children {
                            if *slot == Some(*cidx) {
                                *slot = None;
                                break;
                            }
                        }
                    }

                    // Parent still has remaining children — keep it in the reducible list
                    self.reducible[level].push(node_idx);
                } else {
                    // Can't reach target_k even merging all children; do full merge and
                    // accept fewer than k leaves.
                    self.merge_children_into(node_idx, &child_indices, true);
                    self.leaf_count = after_full;
                }
            }
        }
    }

    /// Merge a set of children into the parent node.
    /// If `make_leaf` is true the parent becomes a leaf and all child slots are
    /// cleared; otherwise the parent stays internal (for partial merges).
    fn merge_children_into(&mut self, node_idx: usize, child_indices: &[usize], make_leaf: bool) {
        let mut tc: u64 = 0;
        let mut sr = 0.0;
        let mut sg = 0.0;
        let mut sb = 0.0;
        for &cidx in child_indices {
            let child = &self.nodes[cidx];
            tc += child.pixel_count;
            sr += child.sum_r;
            sg += child.sum_g;
            sb += child.sum_b;
        }
        let node = &mut self.nodes[node_idx];
        node.pixel_count += tc;
        node.sum_r += sr;
        node.sum_g += sg;
        node.sum_b += sb;
        if make_leaf {
            node.children = [None; 8];
            node.is_leaf = true;
        }
    }

    // --- Colour collection (explicit stack — no recursion) ---

    /// Walk the tree and collect every leaf's (mean-coords, pixel_count).
    /// Also returns the mean coords of the leaf with the highest pixel_count
    /// (the dominant colour).
    fn collect_colors_and_dominant(&self) -> (Vec<([f64; 3], u64)>, Option<[f64; 3]>) {
        let mut results = Vec::new();
        let mut max_count: u64 = 0;
        let mut dominant: Option<[f64; 3]> = None;
        let mut stack: Vec<usize> = vec![self.root];

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];
            if node.is_leaf && node.pixel_count > 0 {
                let mean = [
                    node.sum_r / node.pixel_count as f64,
                    node.sum_g / node.pixel_count as f64,
                    node.sum_b / node.pixel_count as f64,
                ];
                if node.pixel_count > max_count {
                    max_count = node.pixel_count;
                    dominant = Some(mean);
                }
                results.push((mean, node.pixel_count));
            } else {
                // Push children onto stack
                for child_opt in &node.children {
                    if let Some(cidx) = child_opt {
                        stack.push(*cidx);
                    }
                }
            }
        }

        (results, dominant)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run(pixels: &[[f64; 3]], cs: ColorSpace, k: usize) -> Result<AlgorithmResult> {
    // Empty input → empty output
    if pixels.is_empty() {
        return Ok(AlgorithmResult {
            palette: vec![],
            dominant: make_entry([0.0, 0.0, 0.0], 0.0),
            duration: std::time::Duration::ZERO,
        });
    }

    let k = k.max(1).min(pixels.len());

    let ((palette, dominant), duration) = timed(|| {
        let mut quantizer = OctreeQuantizer::new();

        // Holds the final (mean-coord, pixel_count) for every leaf.
        let leaf_colors: Vec<([f64; 3], u64)>;
        // Dominant colour in the octree's own coordinate space.
        let dominant_coord: [f64; 3];

        if cs == ColorSpace::RGB {
            // ── Bit-based strategy (RGB) ──────────────────────────────
            for p in pixels {
                let scaled = [p[0] * 255.0, p[1] * 255.0, p[2] * 255.0];
                quantizer.insert_bit(quantizer.root, scaled, 0);
            }
            quantizer.reduce(k);

            let (leaves, dom) = quantizer.collect_colors_and_dominant();
            // Scale coords back from [0, 255] to [0, 1]
            leaf_colors = leaves
                .into_iter()
                .map(|(c, cnt)| ([c[0] / 255.0, c[1] / 255.0, c[2] / 255.0], cnt))
                .collect();
            dominant_coord = match dom {
                Some(c) => [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0],
                None => [0.0, 0.0, 0.0],
            };
        } else {
            // ── Range-based strategy (all other colour spaces) ────────
            let coords = cs.convert_batch_to(pixels);

            // Global bounding box
            let mut min = [f64::MAX; 3];
            let mut max = [f64::MIN; 3];
            for c in &coords {
                for i in 0..3 {
                    if c[i] < min[i] {
                        min[i] = c[i];
                    }
                    if c[i] > max[i] {
                        max[i] = c[i];
                    }
                }
            }
            // Prevent zero-width dimensions (would cause degenerate splits)
            for i in 0..3 {
                if (max[i] - min[i]).abs() < 1e-10 {
                    max[i] = min[i] + 1.0;
                }
            }

            for &c in &coords {
                quantizer.insert_range(quantizer.root, c, 0, min, max);
            }
            quantizer.reduce(k);

            let (leaves, dom) = quantizer.collect_colors_and_dominant();
            leaf_colors = leaves;
            dominant_coord = dom.unwrap_or([0.0, 0.0, 0.0]);
        }

        // ── Convert leaf coords back to [0, 1] RGB ────────────────────
        let coords_only: Vec<[f64; 3]> = leaf_colors.iter().map(|(c, _)| *c).collect();
        let palette_rgb: Vec<[f64; 3]> = if cs == ColorSpace::RGB {
            coords_only // already [0, 1] RGB
        } else {
            cs.convert_batch_from(&coords_only)
        };

        // Dominant colour → RGB
        let dominant_rgb: [f64; 3] = if cs == ColorSpace::RGB {
            dominant_coord
        } else {
            cs.convert_from(dominant_coord)
        };

        // ── Build PaletteEntry list ───────────────────────────────────
        let total_in_leaves: f64 = leaf_colors.iter().map(|(_, cnt)| *cnt as f64).sum();
        let total_in_leaves = if total_in_leaves > 0.0 {
            total_in_leaves
        } else {
            1.0
        };

        let mut palette: Vec<PaletteEntry> = palette_rgb
            .iter()
            .zip(leaf_colors.iter())
            .map(|(rgb, (_, cnt))| make_entry(*rgb, *cnt as f64 / total_in_leaves))
            .collect();

        let dominant = make_entry(
            dominant_rgb,
            leaf_colors
                .iter()
                .map(|(_, cnt)| *cnt as f64)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0)
                / total_in_leaves,
        );

        sort_by_lightness(&mut palette);

        (palette, dominant)
    });

    Ok(AlgorithmResult {
        palette,
        dominant,
        duration,
    })
}
