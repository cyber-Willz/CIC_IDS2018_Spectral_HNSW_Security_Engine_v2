//! # Dynamic Laplacian Regularizer
//!
//! Addresses the **disconnected-graph problem** in online security monitoring:
//! when a physical network segments (link failure, isolation event, or simply
//! a sparse observation window), the graph Laplacian becomes singular
//! (λ₂ = 0), making spectral distances undefined and Fiedler-based anomaly
//! scoring collapse to zero.
//!
//! ## How it works
//!
//! 1. **Physical Laplacian check** — every tick, compute λ₂ on the observed
//!    physical edges only.  If λ₂ < 0.05 the graph is considered fragile /
//!    disconnected.
//!
//! 2. **Virtual edge injection** — for all unconnected pairs, score
//!    *infrastructure similarity* (shared gateway > shared subnet > AS-path
//!    Jaccard).  Pairs above `similarity_threshold` receive a virtual edge
//!    weighted `α · similarity`.
//!
//! 3. **Regularised Laplacian** — rebuild L with both physical and virtual
//!    edges plus an `εI` diagonal shift (shifts all eigenvalues up by ε,
//!    resolving the λ₂=0 singularity on truly isolated nodes).
//!
//! 4. **Propagation gradient** — exposes `propagation_gradient(u)` for use
//!    by the `SpectralSecurityGraph` blast-radius computation; includes
//!    virtual-edge contributions so structural signal flows across otherwise
//!    disconnected components.
//!
//! 5. **Decay** — when physical connectivity recovers (λ₂ ≥ 0.05), virtual
//!    edges expire after 10 ticks so the graph self-heals gracefully.

use std::collections::{HashMap, HashSet};

// ── Public types ──────────────────────────────────────────────────────────────

/// A virtual edge candidate scored by infrastructure similarity.
#[derive(Debug, Clone)]
pub struct VirtualEdge {
    pub src:          u32,
    pub dst:          u32,
    /// Effective weight: α · similarity(src, dst).
    pub weight:       f32,
    pub reason:       VirtualEdgeReason,
    /// Simulation tick at which this edge was injected.
    pub injected_at:  u64,
}

/// Explains why a virtual edge was created.
#[derive(Debug, Clone)]
pub enum VirtualEdgeReason {
    SharedSubnet       { prefix_len: u8 },
    CommonGateway      { gateway_id: u32 },
    AsPathOverlap      { overlap_score: f32 },
    InfrastructureParent { parent_node: u32 },
}

/// Infrastructure metadata used for similarity scoring.
#[derive(Debug, Clone)]
pub struct NodeMeta {
    /// Subnet base address, e.g. `0xC0A80100` = 192.168.1.0.
    pub subnet_prefix: u32,
    pub prefix_len:    u8,
    pub gateway:       Option<u32>,
    /// BGP / AS-path hop list.
    pub as_path:       Vec<u32>,
    pub infra_parent:  Option<u32>,
}

/// Summary produced by each [`DynamicLaplacianRegularizer::tick`] call.
#[derive(Debug, Clone)]
pub struct RegularizationReport {
    pub tick:                       u64,
    /// λ₂ on the physical graph alone (may be 0 for disconnected graphs).
    pub lambda2_raw:                f32,
    /// λ₂ after regularisation (always > 0 if ε > 0).
    pub lambda2_regularized:        f32,
    /// Number of virtual edges injected this tick.
    pub virtual_edges_injected:     usize,
    /// True when the regulariser is actively bridging a disconnected topology.
    pub regularizer_active:         bool,
    /// True when the regularised λ₂ is large enough for valid gradient flow.
    pub propagation_gradient_valid: bool,
}

// ── Core engine ───────────────────────────────────────────────────────────────

/// Online Laplacian regulariser for security graph spectral analysis.
///
/// # Usage
///
/// ```rust,no_run
/// use sec_net_engine::laplacian_regularizer::{DynamicLaplacianRegularizer, NodeMeta};
///
/// let mut reg = DynamicLaplacianRegularizer::new(4, 0.01, 0.5);
/// reg.update_physical_edges(vec![(0, 1, 1.0), (2, 3, 1.0)]); // two disconnected pairs
/// let report = reg.tick(0);
/// assert!(report.regularizer_active);          // λ₂ < 0.05 — regulariser fires
/// assert!(report.propagation_gradient_valid);  // λ₂_reg > 1e-4
/// ```
pub struct DynamicLaplacianRegularizer {
    n:                  usize,
    physical_edges:     Vec<(u32, u32, f32)>,
    virtual_edges:      Vec<VirtualEdge>,
    node_meta:          HashMap<u32, NodeMeta>,
    /// λ₂ from the last [`tick`] (regularised).
    pub lambda2:        f32,
    /// Identity regularisation coefficient ε (default: 0.01).
    pub epsilon:        f32,
    /// Virtual-edge weight scaling coefficient α (default: 0.5).
    pub alpha:          f32,
    /// Minimum similarity score for virtual edge injection (default: 0.4).
    pub similarity_threshold: f32,
    /// Fiedler vector from the last [`tick`].
    pub fiedler_vec:    Vec<f32>,
    lambda2_history:    Vec<f32>,
}

impl DynamicLaplacianRegularizer {
    /// Create a new regulariser for a graph of `n` nodes.
    pub fn new(n: usize, epsilon: f32, alpha: f32) -> Self {
        Self {
            n,
            physical_edges:     Vec::new(),
            virtual_edges:      Vec::new(),
            node_meta:          HashMap::new(),
            lambda2:            0.0,
            epsilon,
            alpha,
            similarity_threshold: 0.4,
            fiedler_vec:        vec![0.0; n],
            lambda2_history:    Vec::new(),
        }
    }

    /// Register infrastructure metadata for node `id`.
    pub fn register_node(&mut self, id: u32, meta: NodeMeta) {
        self.node_meta.insert(id, meta);
    }

    /// Replace the current physical edge set (full snapshot, not incremental).
    pub fn update_physical_edges(&mut self, edges: Vec<(u32, u32, f32)>) {
        self.physical_edges = edges;
    }

    /// Return a reference to all currently active virtual edges.
    pub fn virtual_edges(&self) -> &[VirtualEdge] {
        &self.virtual_edges
    }

    /// Sliding-window trend of raw λ₂ (up to the last 128 ticks).
    pub fn lambda2_trend(&self) -> &[f32] {
        &self.lambda2_history
    }

    // ── Similarity scoring ────────────────────────────────────────────────────

    /// Compute structural similarity between nodes `u` and `v` in [0.0, 1.0].
    ///
    /// Priority order: common-gateway (0.9) > infra-parent (0.8) >
    /// subnet-prefix (0.0–1.0) > AS-path Jaccard × 0.6.
    fn similarity(&self, u: u32, v: u32) -> (f32, VirtualEdgeReason) {
        let default_reason = VirtualEdgeReason::SharedSubnet { prefix_len: 0 };

        let meta_u = match self.node_meta.get(&u) {
            Some(m) => m,
            None => return (0.0, default_reason),
        };
        let meta_v = match self.node_meta.get(&v) {
            Some(m) => m,
            None => return (0.0, default_reason),
        };

        // Common gateway — strongest structural signal
        if let (Some(gw_u), Some(gw_v)) = (meta_u.gateway, meta_v.gateway) {
            if gw_u == gw_v {
                return (0.9, VirtualEdgeReason::CommonGateway { gateway_id: gw_u });
            }
        }

        // Shared infrastructure parent (e.g. same rack / hypervisor)
        if let (Some(p_u), Some(p_v)) = (meta_u.infra_parent, meta_v.infra_parent) {
            if p_u == p_v {
                return (
                    0.8,
                    VirtualEdgeReason::InfrastructureParent { parent_node: p_u },
                );
            }
        }

        // Subnet prefix match — score proportional to shared prefix length
        let mask_u: u64 = !0u64 << (32 - meta_u.prefix_len as u64);
        let mask_v: u64 = !0u64 << (32 - meta_v.prefix_len as u64);
        let common_mask = mask_u & mask_v;
        if (meta_u.subnet_prefix as u64 & common_mask)
            == (meta_v.subnet_prefix as u64 & common_mask)
        {
            let shared_bits = common_mask.leading_ones() as u8;
            let score = shared_bits as f32 / 32.0;
            return (score, VirtualEdgeReason::SharedSubnet { prefix_len: shared_bits });
        }

        // AS-path Jaccard overlap
        let set_u: HashSet<u32> = meta_u.as_path.iter().cloned().collect();
        let set_v: HashSet<u32> = meta_v.as_path.iter().cloned().collect();
        let inter = set_u.intersection(&set_v).count();
        let union = set_u.union(&set_v).count();
        if union > 0 {
            let jaccard = inter as f32 / union as f32;
            return (
                jaccard * 0.6,
                VirtualEdgeReason::AsPathOverlap { overlap_score: jaccard },
            );
        }

        (0.0, default_reason)
    }

    // ── Laplacian builders ────────────────────────────────────────────────────

    fn build_regularized_laplacian_physical_only(&self) -> Vec<Vec<f32>> {
        let n = self.n;
        let mut l = vec![vec![0.0f32; n]; n];
        for &(u, v, w) in &self.physical_edges {
            let (ui, vi) = (u as usize, v as usize);
            l[ui][ui] += w;
            l[vi][vi] += w;
            l[ui][vi] -= w;
            l[vi][ui] -= w;
        }
        for i in 0..n {
            l[i][i] += self.epsilon;
        }
        l
    }

    /// Build **L + εI + Σ wₖ·lₖ** (physical + virtual edges + identity shift).
    fn build_regularized_laplacian(&self) -> Vec<Vec<f32>> {
        let n = self.n;
        let mut l = vec![vec![0.0f32; n]; n];

        for &(u, v, w) in &self.physical_edges {
            let (ui, vi) = (u as usize, v as usize);
            l[ui][ui] += w;
            l[vi][vi] += w;
            l[ui][vi] -= w;
            l[vi][ui] -= w;
        }

        // Each virtual edge (u,v,w) contributes w·(eᵤ − eᵥ)(eᵤ − eᵥ)ᵀ
        for ve in &self.virtual_edges {
            let (ui, vi) = (ve.src as usize, ve.dst as usize);
            l[ui][ui] += ve.weight;
            l[vi][vi] += ve.weight;
            l[ui][vi] -= ve.weight;
            l[vi][ui] -= ve.weight;
        }

        // εI — shifts all eigenvalues up by ε
        for i in 0..n {
            l[i][i] += self.epsilon;
        }
        l
    }

    // ── Eigenvalue computation ────────────────────────────────────────────────

    /// Shift-invert power iteration to extract the Fiedler value (λ₂) and
    /// the corresponding Fiedler vector.
    ///
    /// The constant eigenvector (λ₁ ≈ 0, direction 1/√n) is projected out
    /// before each iteration so the method converges to the *second* smallest
    /// eigenvalue.
    fn fiedler_power_iter(&self, l: &[Vec<f32>], iters: usize) -> (f32, Vec<f32>) {
        let n = self.n;
        let inv_sqrt_n = 1.0 / (n as f32).sqrt();

        // Deterministic starting vector
        let mut v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37 + 1.0).sin()).collect();

        let project_out = |vec: &mut Vec<f32>| {
            let dot: f32 = vec.iter().sum::<f32>() * inv_sqrt_n;
            for x in vec.iter_mut() {
                *x -= dot * inv_sqrt_n;
            }
        };

        let normalize = |vec: &mut Vec<f32>| {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in vec.iter_mut() {
                    *x /= norm;
                }
            }
        };

        project_out(&mut v);
        normalize(&mut v);

        // Shift σ chosen above the expected spectral radius so (σI − L) is SPD
        let sigma = 10.0f32;
        for _ in 0..iters {
            let mut w: Vec<f32> = (0..n)
                .map(|i| {
                    sigma * v[i]
                        - l[i]
                            .iter()
                            .zip(v.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                })
                .collect();
            project_out(&mut w);
            normalize(&mut w);
            v = w;
        }

        // Rayleigh quotient vᵀLv / vᵀv
        let lv: Vec<f32> = (0..n)
            .map(|i| l[i].iter().zip(v.iter()).map(|(a, b)| a * b).sum())
            .collect();
        let rayleigh: f32 = v.iter().zip(lv.iter()).map(|(a, b)| a * b).sum::<f32>();

        (rayleigh.max(0.0), v)
    }

    // ── Main tick ─────────────────────────────────────────────────────────────

    /// Advance the regulariser by one monitoring tick.
    ///
    /// 1. Compute λ₂ on physical edges.
    /// 2. If λ₂ < 0.05 — inject virtual edges for similar node pairs.
    /// 3. Recompute λ₂ on the regularised graph.
    /// 4. Record λ₂ history and return a [`RegularizationReport`].
    pub fn tick(&mut self, current_tick: u64) -> RegularizationReport {
        // Phase 1: physical graph connectivity check
        let l_phys = self.build_regularized_laplacian_physical_only();
        let (lambda2_raw, _) = self.fiedler_power_iter(&l_phys, 60);

        // Phase 2: virtual edge injection / decay
        let mut injected = 0usize;
        if lambda2_raw < 0.05 {
            self.virtual_edges.clear();
            let n = self.n as u32;
            for u in 0..n {
                for v in (u + 1)..n {
                    let already_physical = self
                        .physical_edges
                        .iter()
                        .any(|&(a, b, _)| (a == u && b == v) || (a == v && b == u));
                    if already_physical {
                        continue;
                    }
                    let (sim, reason) = self.similarity(u, v);
                    if sim >= self.similarity_threshold {
                        self.virtual_edges.push(VirtualEdge {
                            src:         u,
                            dst:         v,
                            weight:      self.alpha * sim,
                            reason,
                            injected_at: current_tick,
                        });
                        injected += 1;
                    }
                }
            }
        } else {
            // Decay: expire virtual edges when physical connectivity recovers
            self.virtual_edges.retain(|ve| {
                current_tick.saturating_sub(ve.injected_at) < 10
            });
        }

        // Phase 3: full regularised graph
        let l_reg = self.build_regularized_laplacian();
        let (lambda2_reg, fiedler) = self.fiedler_power_iter(&l_reg, 80);
        self.lambda2     = lambda2_reg;
        self.fiedler_vec = fiedler;

        self.lambda2_history.push(lambda2_raw);
        if self.lambda2_history.len() > 128 {
            self.lambda2_history.remove(0);
        }

        RegularizationReport {
            tick:                       current_tick,
            lambda2_raw,
            lambda2_regularized:        lambda2_reg,
            virtual_edges_injected:     injected,
            regularizer_active:         lambda2_raw < 0.05,
            propagation_gradient_valid: lambda2_reg > 1e-4,
        }
    }

    // ── Gradient ─────────────────────────────────────────────────────────────

    /// Propagation gradient for node `u` using the current Fiedler coordinates.
    ///
    /// For each physical and virtual neighbour `v`:
    /// `∇(u, v) = w(u,v) · (f[u] − f[v])`
    ///
    /// Large positive values indicate flow *away* from `u`; negative values
    /// indicate flow *toward* `u`.  Used by the blast-radius scorer to rank
    /// lateral movement risk.
    pub fn propagation_gradient(&self, u: usize) -> Vec<(usize, f32)> {
        let fv = &self.fiedler_vec;
        let mut gradients = Vec::new();

        for &(src, dst, w) in &self.physical_edges {
            let (s, d) = (src as usize, dst as usize);
            if s == u || d == u {
                let neighbor = if s == u { d } else { s };
                gradients.push((neighbor, w * (fv[u] - fv[neighbor])));
            }
        }

        // Virtual edges also carry structural signal — include them
        for ve in &self.virtual_edges {
            let (s, d) = (ve.src as usize, ve.dst as usize);
            if s == u || d == u {
                let neighbor = if s == u { d } else { s };
                gradients.push((neighbor, ve.weight * (fv[u] - fv[neighbor])));
            }
        }

        gradients
    }
}