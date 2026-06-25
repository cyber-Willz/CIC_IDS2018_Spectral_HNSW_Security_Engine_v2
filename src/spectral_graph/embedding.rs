use log::{debug, warn};
use serde::{Deserialize, Serialize};

use super::{
    error::{GraphError, GraphResult},
    graph::Graph,
};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Tuning knobs for the Jacobi cyclic eigendecomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JacobiConfig {
    /// Maximum Jacobi sweeps before giving up.
    pub max_iterations: usize,
    /// Off-diagonal convergence threshold.
    pub epsilon:        f64,
    /// Number of eigenvectors to retain (`None` = all).
    pub dimensions:     Option<usize>,
}

impl Default for JacobiConfig {
    fn default() -> Self {
        JacobiConfig { max_iterations: 1_000, epsilon: 1e-10, dimensions: None }
    }
}

// ── SpectralEmbedding ─────────────────────────────────────────────────────────

/// Continuous spectral-space representation of a discrete graph.
///
/// Each node maps to row `i` in `eigenvector_matrix`; columns are sorted by
/// ascending eigenvalue (λ₀ ≈ 0 trivial, λ₁ = Fiedler value).
///
/// # Distance functions
///
/// | Method | Formula | Use case |
/// |---|---|---|
/// | [`geometric_distance`] | Commute-time (resistance) | Topology-aware anomaly ranking |
/// | [`euclidean_distance`] | Plain L2 | Diagnostics |
/// | [`fiedler_distance`] | 1-D Fiedler projection | Fast cluster-split check |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralEmbedding {
    pub num_nodes:              usize,
    pub num_dims:               usize,
    /// Row `i` = eigenvector coordinates for node `i`.
    pub eigenvector_matrix:     Vec<Vec<f64>>,
    /// Eigenvalues sorted in ascending order (λ₀ ≈ 0, λ₁ = Fiedler, …).
    pub eigenvalues:            Vec<f64>,
    /// Algebraic connectivity λ₁.  Zero iff the graph is disconnected.
    pub algebraic_connectivity: f64,
}

impl SpectralEmbedding {
    /// Compute the spectral embedding of `graph` using the Jacobi
    /// eigendecomposition of its unnormalised Laplacian.
    pub fn embed(graph: &Graph, cfg: &JacobiConfig) -> GraphResult<Self> {
        let n = graph.num_nodes();
        if n < 2 {
            return Err(GraphError::InsufficientNodesForEmbedding(n));
        }

        let mut l = graph.laplacian();
        // Accumulate rotation matrices into U (starts as identity).
        let mut u: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();

        let mut converged = false;
        for iter in 0..cfg.max_iterations {
            let (max_val, p, q) = off_diagonal_max(&l, n);
            if max_val < cfg.epsilon {
                debug!("Jacobi converged after {iter} iterations (residual={max_val:.2e})");
                converged = true;
                break;
            }
            let (c, s) = jacobi_cs(l[p][p], l[q][q], l[p][q]);
            apply_jacobi_rotation_to_l(&mut l, n, p, q, c, s);
            for i in 0..n {
                let u_ip = u[i][p];
                let u_iq = u[i][q];
                u[i][p] = c * u_ip - s * u_iq;
                u[i][q] = s * u_ip + c * u_iq;
            }
        }

        if !converged {
            let residual = off_diagonal_max(&l, n).0;
            warn!("Jacobi did not fully converge (residual={residual:.2e})");
            if residual > 1e-4 {
                return Err(GraphError::EigenConvergenceFailed {
                    max_iterations: cfg.max_iterations,
                    residual,
                });
            }
        }

        // The graph Laplacian is positive semi-definite; any negative diagonal
        // value after Jacobi is floating-point noise — clamp to zero.
        let mut pairs: Vec<(f64, usize)> =
            (0..n).map(|i| (l[i][i].max(0.0), i)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let num_dims = cfg.dimensions.unwrap_or(n).min(n);
        let eigenvalues: Vec<f64> =
            pairs[..num_dims].iter().map(|&(ev, _)| ev).collect();
        let eigenvector_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| pairs[..num_dims].iter().map(|&(_, c)| u[i][c]).collect())
            .collect();

        let algebraic_connectivity = if num_dims > 1 { eigenvalues[1] } else { 0.0 };

        Ok(SpectralEmbedding {
            num_nodes: n,
            num_dims,
            eigenvector_matrix,
            eigenvalues,
            algebraic_connectivity,
        })
    }

    // ── Distance metrics ──────────────────────────────────────────────────────

    /// **Commute-time (resistance) distance** — weights each dim `k` by `1/λₖ`.
    ///
    /// ```text
    /// d_CT(u, v) = sqrt( Σₖ (φₖ(u) − φₖ(v))² / λₖ )   (k ≥ 1, λₖ > 0)
    /// ```
    pub fn geometric_distance(&self, u: usize, v: usize) -> GraphResult<f64> {
        if self.num_dims < 2 {
            return Err(GraphError::InsufficientNodesForEmbedding(self.num_dims));
        }
        self.validate_node(u)?;
        self.validate_node(v)?;
        let dist = (1..self.num_dims)
            .filter_map(|k| {
                let lambda = self.eigenvalues[k];
                if lambda < 1e-10 {
                    return None;
                }
                let d = self.eigenvector_matrix[u][k] - self.eigenvector_matrix[v][k];
                Some((d * d) / lambda)
            })
            .sum::<f64>()
            .sqrt();
        Ok(dist)
    }

    /// Raw Euclidean distance in eigenvector space.  Use for diagnostics only;
    /// collapses on symmetric graphs.
    pub fn euclidean_distance(&self, u: usize, v: usize) -> GraphResult<f64> {
        self.validate_node(u)?;
        self.validate_node(v)?;
        let dist = (0..self.num_dims)
            .map(|k| {
                let d = self.eigenvector_matrix[u][k] - self.eigenvector_matrix[v][k];
                d * d
            })
            .sum::<f64>()
            .sqrt();
        Ok(dist)
    }

    /// Fiedler-only 1-D distance — fast structural-split approximation.
    ///
    /// On disconnected graphs λ₁ = 0 and the standard Fiedler vector gives
    /// zero separation.  This implementation scans forward to the first
    /// eigenvector dimension with λₖ > 1e-10 so it remains informative even
    /// on partially-disconnected topologies.
    pub fn fiedler_distance(&self, u: usize, v: usize) -> GraphResult<f64> {
        if self.num_dims < 2 {
            return Err(GraphError::InsufficientNodesForEmbedding(self.num_dims));
        }
        self.validate_node(u)?;
        self.validate_node(v)?;
        let first_nontrivial = self
            .eigenvalues
            .iter()
            .position(|&ev| ev > 1e-10)
            .unwrap_or(1);
        Ok((self.eigenvector_matrix[u][first_nontrivial]
            - self.eigenvector_matrix[v][first_nontrivial])
            .abs())
    }

    /// Coordinate of `node` along eigenvector dimension `dim`.
    pub fn coordinate(&self, node: usize, dim: usize) -> GraphResult<f64> {
        self.validate_node(node)?;
        if dim >= self.num_dims {
            return Err(GraphError::DimensionOutOfRange(dim, self.num_dims));
        }
        Ok(self.eigenvector_matrix[node][dim])
    }

    /// The full Fiedler vector (column 1 of the eigenvector matrix).
    pub fn fiedler_vector(&self) -> GraphResult<Vec<f64>> {
        if self.num_dims < 2 {
            return Err(GraphError::InsufficientNodesForEmbedding(self.num_dims));
        }
        Ok((0..self.num_nodes)
            .map(|i| self.eigenvector_matrix[i][1])
            .collect())
    }

    /// Extract the embedding coordinates for a single node as a flat `Vec<f32>`.
    /// Used when appending spectral features to the anomaly encoder input.
    pub fn node_coords_f32(&self, node: usize) -> GraphResult<Vec<f32>> {
        self.validate_node(node)?;
        Ok(self.eigenvector_matrix[node]
            .iter()
            .map(|&x| x as f32)
            .collect())
    }

    #[inline]
    fn validate_node(&self, idx: usize) -> GraphResult<()> {
        if idx >= self.num_nodes {
            Err(GraphError::NodeOutOfRange(idx, self.num_nodes))
        } else {
            Ok(())
        }
    }
}

// ── Jacobi helpers ────────────────────────────────────────────────────────────

#[inline]
fn off_diagonal_max(l: &[Vec<f64>], n: usize) -> (f64, usize, usize) {
    let (mut max_val, mut p, mut q) = (0.0_f64, 0, 1);
    for i in 0..n {
        for j in (i + 1)..n {
            let v = l[i][j].abs();
            if v > max_val {
                max_val = v;
                p = i;
                q = j;
            }
        }
    }
    (max_val, p, q)
}

#[inline]
fn jacobi_cs(l_pp: f64, l_qq: f64, l_pq: f64) -> (f64, f64) {
    let theta = (l_qq - l_pp) / (2.0 * l_pq);
    let t = if theta >= 0.0 {
        1.0 / (theta + (theta * theta + 1.0).sqrt())
    } else {
        -1.0 / (-theta + (theta * theta + 1.0).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    (c, t * c)
}

fn apply_jacobi_rotation_to_l(
    l: &mut Vec<Vec<f64>>,
    n: usize,
    p: usize,
    q: usize,
    c: f64,
    s: f64,
) {
    let (l_pp, l_qq, l_pq) = (l[p][p], l[q][q], l[p][q]);
    l[p][p] = c * c * l_pp - 2.0 * s * c * l_pq + s * s * l_qq;
    l[q][q] = s * s * l_pp + 2.0 * s * c * l_pq + c * c * l_qq;
    l[p][q] = 0.0;
    l[q][p] = 0.0;
    for i in 0..n {
        if i != p && i != q {
            let (l_ip, l_iq) = (l[i][p], l[i][q]);
            l[i][p] = c * l_ip - s * l_iq;
            l[p][i] = l[i][p];
            l[i][q] = s * l_ip + c * l_iq;
            l[q][i] = l[i][q];
        }
    }
}