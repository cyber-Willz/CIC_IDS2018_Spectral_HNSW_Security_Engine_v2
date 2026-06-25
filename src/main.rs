//!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//! 
//! 
//! //! # CIC-IDS2018 × Spectral Security Engine  v2
//!
//! ## Architecture
//!
//! ```text
//! CSV rows (CicRow)
//!       │
//!       ▼
//! embed_cic_numeric()  ──►  40-dim L2-normalised feature vector
//!       │
//!       ├──► AnomalyModel (Burn autoencoder)  ──►  MSE loss → Severity
//!       │
//!       ├──► HnswFlowIndex  ──►  k-NN similar-flow lookup
//!       │
//!       ├──► Qdrant collection  ──►  persistent vector store + metadata
//!       │
//!       └──► SpectralSecurityGraph  ──►  Jacobi eigendecomp → CT / Fiedler distances
//! ```
//!
//! ## Bug-fixes vs original v1
//!
//! | # | Bug | Fix |
//! |---|-----|-----|
//! | 1 | Dimension mismatch panic — `ModelConfig` defaulted to `input_dim=32` but `FEATURE_DIM=40` | `pretrain_on_benign` calls `.with_input_dim(FEATURE_DIM)` |
//! | 2 | `to_vec()` turbofish missing → runtime type ambiguity | Explicit `to_vec::<f32>()` |
//! | 3 | Duplicate `GraphError` in `lib.rs` and `error.rs` | Canonical definition lives only in `error.rs` |
//! | 4 | Reshape used wrong constant | Fixed with correct `FEATURE_DIM` |

pub mod laplacian_regularizer;
pub mod spectral_graph;

use laplacian_regularizer::{
    DynamicLaplacianRegularizer, NodeMeta, RegularizationReport, VirtualEdgeReason,
};

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{error, info, warn};

// ── Qdrant ────────────────────────────────────────────────────────────────────
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    Distance, FieldType, Filter, PointStruct, ScrollPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant, QdrantError};

// ── Burn (CPU / NdArray) ──────────────────────────────────────────────────────
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::config::Config;
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Shape, Tensor, TensorData};

// ── HNSW ──────────────────────────────────────────────────────────────────────
use anndists::dist::DistL2;
use hnsw_rs::prelude::*;

// ── Spectral ──────────────────────────────────────────────────────────────────
use spectral_graph::{
    embedding::{JacobiConfig, SpectralEmbedding},
    graph::Graph,
    report::GraphReport,
};

// ============================================================================
// Constants
// ============================================================================

/// 40-dimensional feature vector: 32 continuous (log1p-scaled) + 8
/// application-layer binary/categorical flags.  Single source of truth for
/// model width, HNSW index width, and Qdrant vector dimension.
pub const FEATURE_DIM: usize = 40;

/// Autoencoder bottleneck width.
const LATENT_DIM: usize = 20;

const COLLECTION_NAME: &str = "cic_ids2018_v2";
const HNSW_KNN: usize = 5;

// ============================================================================
// Error
// ============================================================================

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Qdrant: {0}")]
    Qdrant(#[from] QdrantError),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("Spectral: {0}")]
    Spectral(#[from] spectral_graph::error::GraphError),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Engine: {0}")]
    Engine(String),
}

// ============================================================================
// Domain types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum EdgeType {
    ObservedWith            = 1,
    AuthenticatedTo         = 2,
    CommunicatedWith        = 3,
    ReverseCommunicatedWith = 4,
}

impl EdgeType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ObservedWith            => "ObservedWith",
            Self::AuthenticatedTo         => "AuthenticatedTo",
            Self::CommunicatedWith        => "CommunicatedWith",
            Self::ReverseCommunicatedWith => "ReverseCommunicatedWith",
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum Entity {
    User(String),
    Host(String),
    IpAddress(String),
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(s)      => write!(f, "User:{s}"),
            Self::Host(s)      => write!(f, "Host:{s}"),
            Self::IpAddress(s) => write!(f, "Ip:{s}"),
        }
    }
}

// ============================================================================
// Severity
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

impl Severity {
    pub fn from_loss(loss: f32, threshold: f32) -> Self {
        if loss < threshold {
            Self::Info
        } else if loss < threshold * 1.5 {
            Self::Low
        } else if loss < threshold * 3.0 {
            Self::Medium
        } else if loss < threshold * 6.0 {
            Self::High
        } else {
            Self::Critical
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Info     => "INFO",
                Self::Low      => "LOW",
                Self::Medium   => "MEDIUM",
                Self::High     => "HIGH",
                Self::Critical => "CRITICAL",
            }
        )
    }
}

// ============================================================================
// MITRE ATT&CK
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitreTactic {
    pub tactic_id:     String,
    pub tactic_name:   String,
    pub technique:     String,
    pub trigger_token: String,
}

impl fmt::Display for MitreTactic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} — {} (trigger: '{}')",
            self.tactic_id, self.tactic_name, self.technique, self.trigger_token
        )
    }
}

static MITRE_VOCAB: &[(&str, &str, &str, &str)] = &[
    ("exploit",     "TA0001", "Initial Access",      "T1190 – Exploit Public-Facing App"),
    ("overflow",    "TA0002", "Execution",            "T1203 – Exploitation for Client Execution"),
    ("rce",         "TA0002", "Execution",            "T1203 – Remote Code Execution"),
    ("shellcode",   "TA0002", "Execution",            "T1059 – Command Scripting Interpreter"),
    ("escalation",  "TA0004", "Privilege Escalation", "T1068 – Exploitation for Priv Escalation"),
    ("sudo",        "TA0004", "Privilege Escalation", "T1548 – Abuse Elevation Control Mechanism"),
    ("lateral",     "TA0008", "Lateral Movement",     "T1021 – Remote Services"),
    ("mimikatz",    "TA0006", "Credential Access",    "T1003 – OS Credential Dumping"),
    ("credential",  "TA0006", "Credential Access",    "T1552 – Unsecured Credentials"),
    ("exfil",       "TA0010", "Exfiltration",         "T1041 – Exfiltration Over C2 Channel"),
    ("c2",          "TA0011", "Command and Control",  "T1071 – Application Layer Protocol"),
    ("beacon",      "TA0011", "Command and Control",  "T1071 – Application Layer Protocol"),
    ("ransomware",  "TA0040", "Impact",               "T1486 – Data Encrypted for Impact"),
    ("ddos",        "TA0040", "Impact",               "T1498 – Network Denial of Service"),
    ("dos",         "TA0040", "Impact",               "T1499 – Endpoint Denial of Service"),
    ("injection",   "TA0002", "Execution",            "T1055 – Process Injection"),
    ("persistence", "TA0003", "Persistence",          "T1053 – Scheduled Task/Job"),
    ("backdoor",    "TA0003", "Persistence",          "T1505 – Server Software Component"),
    ("cve",         "TA0001", "Initial Access",       "T1190 – Known Vulnerability Exploitation"),
    ("brute",       "TA0006", "Credential Access",    "T1110 – Brute Force"),
    ("spray",       "TA0006", "Credential Access",    "T1110 – Password Spraying"),
    ("phish",       "TA0001", "Initial Access",       "T1566 – Phishing"),
    ("scan",        "TA0043", "Reconnaissance",       "T1046 – Network Service Scanning"),
    ("recon",       "TA0043", "Reconnaissance",       "T1595 – Active Scanning"),
    ("obfuscat",    "TA0005", "Defense Evasion",      "T1027 – Obfuscated Files or Information"),
    ("rootkit",     "TA0005", "Defense Evasion",      "T1014 – Rootkit"),
    ("bypass",      "TA0005", "Defense Evasion",      "T1562 – Impair Defenses"),
    ("infilter",    "TA0008", "Lateral Movement",     "T1021 – Remote Services (Infiltration)"),
    ("hulk",        "TA0040", "Impact",               "T1499 – Endpoint Denial of Service"),
    ("goldeneye",   "TA0040", "Impact",               "T1499 – Endpoint Denial of Service"),
    ("slowhttp",    "TA0040", "Impact",               "T1499 – Application Exhaustion Flood"),
    ("slowloris",   "TA0040", "Impact",               "T1499 – Application Exhaustion Flood"),
    ("loic",        "TA0040", "Impact",               "T1498 – Network Denial of Service"),
    ("heartbleed",  "TA0001", "Initial Access",       "T1190 – OpenSSL Heartbleed CVE-2014-0160"),
    ("xss",         "TA0002", "Execution",            "T1059.007 – Cross-Site Scripting"),
    ("sql",         "TA0001", "Initial Access",       "T1190 – SQL Injection"),
    ("bot",         "TA0011", "Command and Control",  "T1071 – Application Layer Protocol (Botnet)"),
    ("ftpbrute",    "TA0006", "Credential Access",    "T1110 – Brute Force FTP"),
    ("wiper",       "TA0040", "Impact",               "T1485 – Data Destruction"),
    // "infilteration" (CIC dataset typo) contains "infilter" — covered above.
    ("ftp",         "TA0006", "Credential Access",    "T1110 – Brute Force FTP"),
];

/// Map a CIC-IDS2018 label string to the best-matching MITRE ATT&CK tactic.
pub fn classify_mitre_from_label(label: &str) -> Option<MitreTactic> {
    let lower = label.to_lowercase();
    for &(kw, tid, tname, tech) in MITRE_VOCAB {
        if lower.contains(kw) {
            return Some(MitreTactic {
                tactic_id:     tid.to_string(),
                tactic_name:   tname.to_string(),
                technique:     tech.to_string(),
                trigger_token: kw.to_string(),
            });
        }
    }
    None
}

// ============================================================================
// Feature extraction — 40-dimensional vector
// ============================================================================

/// Produce a 40-dimensional, L2-normalised feature vector from a [`CicRow`].
///
/// Dimensions 0–31: continuous network-flow statistics (log1p-scaled).
/// Dimensions 32–39: application-layer categorical/binary flags.
pub fn embed_cic_numeric(row: &CicRow) -> Vec<f32> {
    let continuous: [f64; 32] = [
        row.flow_duration as f64,
        row.tot_fwd_pkts as f64,
        row.tot_bwd_pkts as f64,
        row.totlen_fwd_pkts,
        row.totlen_bwd_pkts,
        row.flow_byts_s,
        row.flow_pkts_s,
        row.fwd_pkts_s,
        row.bwd_pkts_s,
        row.syn_flag_cnt as f64,
        row.rst_flag_cnt as f64,
        row.fin_flag_cnt as f64,
        row.psh_flag_cnt as f64,
        row.ack_flag_cnt as f64,
        row.fwd_pkt_len_max,
        row.bwd_pkt_len_max,
        row.fwd_pkt_len_mean,
        row.bwd_pkt_len_mean,
        row.pkt_len_mean,
        row.pkt_len_std,
        row.init_fwd_win_byts.max(0) as f64,
        row.init_bwd_win_byts.max(0) as f64,
        row.flow_iat_mean,
        row.flow_iat_std,
        row.flow_iat_max,
        row.flow_iat_min,
        row.active_mean,
        row.active_std,
        row.idle_mean,
        row.idle_std,
        row.down_up_ratio,
        row.pkt_size_avg,
    ];

    let port_group: f32 = match (row.protocol, row.dst_port) {
        (_, 80) | (_, 443) | (_, 8080) | (_, 8443) => 1.0,
        (_, 21) | (_, 22) | (_, 23) | (_, 3389)    => 2.0,
        (_, 53)                                     => 3.0,
        (_, p) if p < 1024                          => 4.0,
        _                                           => 0.0,
    };
    let proto_bucket: f32 = match row.protocol {
        6  => 0.0,
        17 => 1.0,
        _  => 2.0,
    };
    let syn_only: f32 =
        if row.syn_flag_cnt > 0 && row.ack_flag_cnt == 0 { 1.0 } else { 0.0 };
    let asym_num = (row.fwd_pkt_len_mean - row.bwd_pkt_len_mean).abs();
    let asym_den = row.fwd_pkt_len_mean + row.bwd_pkt_len_mean + 1.0;
    let payload_asym: f32 = (asym_num / asym_den) as f32;
    let high_freq: f32  = if row.flow_pkts_s > 10_000.0 { 1.0 } else { 0.0 };
    let zero_bwd: f32   = if row.tot_bwd_pkts == 0 { 1.0 } else { 0.0 };
    let large_bwd: f32  = if row.bwd_pkt_len_max > 8_000.0 { 1.0 } else { 0.0 };
    let small_pkt: f32  = if row.pkt_len_mean < 80.0 { 1.0 } else { 0.0 };

    let mut v: Vec<f32> = Vec::with_capacity(FEATURE_DIM);
    v.extend(continuous.iter().map(|&x| (x.max(0.0) + 1.0).ln() as f32));
    v.extend_from_slice(&[
        port_group, proto_bucket, syn_only, payload_asym,
        high_freq,  zero_bwd,    large_bwd, small_pkt,
    ]);
    debug_assert_eq!(v.len(), FEATURE_DIM, "embed_cic_numeric: wrong output length");

    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

// ============================================================================
// Autoencoder — FEATURE_DIM → LATENT_DIM → FEATURE_DIM
// ============================================================================

#[derive(Module, Debug)]
pub struct AnomalyModel<B: Backend> {
    encoder: Linear<B>,
    decoder: Linear<B>,
}

/// Build configuration for [`AnomalyModel`].
///
/// Always call `.with_input_dim(FEATURE_DIM).with_latent_dim(LATENT_DIM)`
/// rather than relying on the struct defaults.
#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 40)]
    pub input_dim:  usize,
    #[config(default = 20)]
    pub latent_dim: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AnomalyModel<B> {
        AnomalyModel {
            encoder: LinearConfig::new(self.input_dim, self.latent_dim).init(device),
            decoder: LinearConfig::new(self.latent_dim, self.input_dim).init(device),
        }
    }
}

impl<B: Backend> AnomalyModel<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::relu(self.encoder.forward(x));
        self.decoder.forward(h)
    }

    /// Returns `(mse_loss, per-dim squared errors)` for a single sample
    /// tensor of shape `[1, FEATURE_DIM]`.
    pub fn score(&self, x: Tensor<B, 2>) -> (f32, Vec<f32>) {
        let recon  = self.forward(x.clone());
        let sq_err = (x - recon).powf_scalar(2.0_f32);
        let flat: Vec<f32> = sq_err
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0_f32; FEATURE_DIM]);
        let mse: f32 = sq_err.mean().into_scalar().elem::<f32>();
        (mse, flat)
    }
}

// ============================================================================
// Pre-training
// ============================================================================

/// Train an autoencoder on benign baseline flows and derive an anomaly
/// threshold from the resulting reconstruction-error distribution.
///
/// Returns `(trained_model, threshold)`.
pub fn pretrain_on_benign(
    benign_rows: &[&CicRow],
    device:      &NdArrayDevice,
    epochs:      usize,
    lr:          f64,
) -> (AnomalyModel<NdArray>, f32) {
    type TB = Autodiff<NdArray>;

    let feature_vecs: Vec<Vec<f32>> =
        benign_rows.iter().map(|r| embed_cic_numeric(r)).collect();
    let n    = feature_vecs.len();
    let flat: Vec<f32> = feature_vecs.iter().flatten().copied().collect();

    let train_tensor = Tensor::<TB, 1>::from_data(
        TensorData::new(flat, Shape::new([n * FEATURE_DIM])),
        device,
    )
    .reshape([n, FEATURE_DIM]);

    let mut model: AnomalyModel<TB> = ModelConfig::new()
        .with_input_dim(FEATURE_DIM)
        .with_latent_dim(LATENT_DIM)
        .init(device);

    let mut optim = AdamConfig::new().init::<TB, AnomalyModel<TB>>();

    for epoch in 0..epochs {
        let recon    = model.forward(train_tensor.clone());
        let loss     = (train_tensor.clone() - recon).powf_scalar(2.0_f32).mean();
        let loss_val: f32 = loss.clone().into_scalar().elem::<f32>();
        let grads    = GradientsParams::from_grads(loss.backward(), &model);
        model        = optim.step(lr, model, grads);
        if epoch % 20 == 0 || epoch == epochs - 1 {
            info!(epoch, loss = loss_val, "Pretraining autoencoder");
        }
    }

    let infer_model: AnomalyModel<NdArray> = model.valid();

    let benign_scores: Vec<f32> = feature_vecs
        .iter()
        .map(|fv| {
            let t = Tensor::<NdArray, 1>::from_data(
                TensorData::new(fv.clone(), Shape::new([FEATURE_DIM])),
                device,
            )
            .unsqueeze::<2>();
            infer_model.score(t).0
        })
        .collect();

    let mean: f32    = benign_scores.iter().sum::<f32>() / n as f32;
    let var: f32     = benign_scores
        .iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f32>()
        / n as f32;
    let std_dev      = var.sqrt();
    let max_benign   = benign_scores.iter().cloned().fold(0.0_f32, f32::max);

    // 3-sigma floor + 1.5× max-benign floor to prevent threshold collapse on
    // small baselines.
    let threshold = (mean + 3.0 * std_dev)
        .max(max_benign * 1.5)
        .max(1e-4);
    info!(mean, std_dev, max_benign, threshold, "Anomaly threshold derived");
    (infer_model, threshold)
}

// ============================================================================
// HNSW index
// ============================================================================

#[derive(Debug, Clone)]
pub struct HnswPoint {
    pub features: Vec<f32>,
    pub label:    String,
    pub src:      String,
    pub dst:      String,
}

pub struct HnswFlowIndex {
    hnsw:   Hnsw<'static, f32, DistL2>,
    points: Vec<HnswPoint>,
}

impl HnswFlowIndex {
    pub fn new() -> Self {
        HnswFlowIndex {
            hnsw:   Hnsw::new(16, 100_000, 16, 200, DistL2 {}),
            points: Vec::new(),
        }
    }

    pub fn insert(&mut self, point: HnswPoint) {
        let id = self.points.len();
        self.hnsw.insert((&point.features, id));
        self.points.push(point);
    }

    pub fn knn(&self, query: &[f32], k: usize) -> Vec<HnswMatch> {
        if self.points.is_empty() {
            return Vec::new();
        }
        self.hnsw
            .search(query, k + 1, 32)
            .into_iter()
            .filter(|n| n.d_id < self.points.len())
            .take(k)
            .map(|n| HnswMatch {
                label:    self.points[n.d_id].label.clone(),
                src:      self.points[n.d_id].src.clone(),
                dst:      self.points[n.d_id].dst.clone(),
                distance: n.distance,
            })
            .collect()
    }
}

impl Default for HnswFlowIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswMatch {
    pub label:    String,
    pub src:      String,
    pub dst:      String,
    pub distance: f32,
}

// ============================================================================
// Spectral metadata
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralMeta {
    pub commute_time_distance:  f64,
    pub fiedler_distance:       f64,
    pub algebraic_connectivity: f64,
    pub spectral_blast_radius:  Vec<SpectralBlastNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralBlastNode {
    pub entity:                String,
    pub commute_time_distance: f64,
    pub fiedler_distance:      f64,
}

// ============================================================================
// Incident report
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentReport {
    pub summary:            String,
    pub severity:           Severity,
    pub source:             String,
    pub destination:        String,
    pub edge_type:          String,
    pub timestamp:          u64,
    pub anomaly_score:      f32,
    pub mitre_tactic:       Option<MitreTactic>,
    pub blast_radius_nodes: Vec<String>,
    pub spectral:           Option<SpectralMeta>,
    pub similar_flows:      Vec<HnswMatch>,
    pub raw_label:          String,
}

impl fmt::Display for IncidentReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "╔══════════════════════════════════════════════════════════════╗"
        )?;
        writeln!(
            f,
            "║  INCIDENT [{sev}]  label={lbl}",
            sev = self.severity,
            lbl = self.raw_label
        )?;
        writeln!(
            f,
            "╠──────────────────────────────────────────────────────────────╣"
        )?;
        writeln!(f, "║  {}", self.summary)?;
        writeln!(
            f,
            "╠──────────────────────────────────────────────────────────────╣"
        )?;
        writeln!(
            f,
            "║  Src  : {}  →  Dst : {}",
            self.source, self.destination
        )?;
        writeln!(
            f,
            "║  Loss : {:.6}  Edge : {}",
            self.anomaly_score, self.edge_type
        )?;
        if let Some(ref t) = self.mitre_tactic {
            writeln!(
                f,
                "╠──────────────────────────────────────────────────────────────╣"
            )?;
            writeln!(f, "║  MITRE : {t}")?;
        }
        if !self.similar_flows.is_empty() {
            writeln!(
                f,
                "╠──────────────────────────────────────────────────────────────╣"
            )?;
            writeln!(
                f,
                "║  HNSW SIMILAR FLOWS ({} neighbours):",
                self.similar_flows.len()
            )?;
            for m in &self.similar_flows {
                writeln!(
                    f,
                    "║    d={:.4}  {}→{}  [{}]",
                    m.distance, m.src, m.dst, m.label
                )?;
            }
        }
        if !self.blast_radius_nodes.is_empty() {
            writeln!(
                f,
                "╠──────────────────────────────────────────────────────────────╣"
            )?;
            writeln!(
                f,
                "║  BFS BLAST RADIUS ({} nodes):",
                self.blast_radius_nodes.len()
            )?;
            for n in &self.blast_radius_nodes {
                writeln!(f, "║    → {n}")?;
            }
        }
        if let Some(ref sp) = self.spectral {
            writeln!(
                f,
                "╠──────────────────────────────────────────────────────────────╣"
            )?;
            writeln!(f, "║  SPECTRAL:")?;
            writeln!(
                f,
                "║    λ₁ = {:.6}  CT-dist={:.6}  Fiedler={:.6}",
                sp.algebraic_connectivity, sp.commute_time_distance, sp.fiedler_distance
            )?;
            if sp.fiedler_distance > 0.3 {
                writeln!(f, "║    ⚠ HIGH Fiedler — cross-cluster lateral movement")?;
            }
            if sp.algebraic_connectivity < 0.1 {
                writeln!(f, "║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk")?;
            }
            writeln!(
                f,
                "║    Spectral blast radius ({} nodes):",
                sp.spectral_blast_radius.len()
            )?;
            for n in sp.spectral_blast_radius.iter().take(5) {
                writeln!(
                    f,
                    "║      CT={:.4}  F={:.4}  → {}",
                    n.commute_time_distance, n.fiedler_distance, n.entity
                )?;
            }
        }
        write!(
            f,
            "╚══════════════════════════════════════════════════════════════╝"
        )
    }
}

// ============================================================================
// SpectralSecurityGraph
// ============================================================================

/// Wraps a [`Graph`] + [`SpectralEmbedding`] with a string-keyed entity index.
pub struct SpectralSecurityGraph {
    graph:          Graph,
    embedding:      SpectralEmbedding,
    entity_index:   Vec<String>,
    entity_to_node: HashMap<String, usize>,
}

impl SpectralSecurityGraph {
    pub fn build(
        edges: &[(String, String)],
        cfg:   &JacobiConfig,
    ) -> Result<Self, spectral_graph::error::GraphError> {
        let mut entity_index:   Vec<String>            = Vec::new();
        let mut entity_to_node: HashMap<String, usize> = HashMap::new();

        let mut intern = |e: &str| -> usize {
            if let Some(&idx) = entity_to_node.get(e) {
                return idx;
            }
            let idx = entity_index.len();
            entity_index.push(e.to_string());
            entity_to_node.insert(e.to_string(), idx);
            idx
        };

        let mut raw_edges: Vec<(usize, usize)> = Vec::with_capacity(edges.len());
        for (src, dst) in edges {
            let u = intern(src);
            let v = intern(dst);
            raw_edges.push((u, v));
        }

        let n     = entity_index.len().max(2);
        let graph = Graph::from_edges(n, &raw_edges)?;
        let embedding = SpectralEmbedding::embed(&graph, cfg)?;

        Ok(Self { graph, embedding, entity_index, entity_to_node })
    }

    pub fn algebraic_connectivity(&self) -> f64 {
        self.embedding.algebraic_connectivity
    }

    /// Returns `(commute_time_distance, fiedler_distance)` for the given pair.
    pub fn pair_distances(&self, src: &str, dst: &str) -> (f64, f64) {
        let (Some(&u), Some(&v)) = (
            self.entity_to_node.get(src),
            self.entity_to_node.get(dst),
        ) else {
            return (0.0, 0.0);
        };
        let ct = self.embedding.geometric_distance(u, v).unwrap_or(0.0);
        let fd = self.embedding.fiedler_distance(u, v).unwrap_or(0.0);
        (ct, fd)
    }

    /// Return the `max_nodes` spectral nearest neighbours of `seed`, sorted by
    /// ascending commute-time distance.
    pub fn spectral_blast_radius(
        &self,
        seed:      &str,
        max_nodes: usize,
    ) -> Vec<SpectralBlastNode> {
        let Some(&seed_idx) = self.entity_to_node.get(seed) else {
            return Vec::new();
        };
        let mut ranked: Vec<SpectralBlastNode> = self
            .entity_index
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != seed_idx)
            .filter_map(|(i, entity)| {
                let ct = self.embedding.geometric_distance(seed_idx, i).ok()?;
                let fd = self.embedding.fiedler_distance(seed_idx, i).ok()?;
                Some(SpectralBlastNode {
                    entity: entity.clone(),
                    commute_time_distance: ct,
                    fiedler_distance: fd,
                })
            })
            .collect();
        ranked.sort_by(|a, b| {
            a.commute_time_distance
                .partial_cmp(&b.commute_time_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(max_nodes);
        ranked
    }

    pub fn report(
        &self,
        pairs: &[(String, String, String)],
    ) -> Result<GraphReport, spectral_graph::error::GraphError> {
        let resolved: Vec<(usize, usize, &str)> = pairs
            .iter()
            .filter_map(|(src, dst, label)| {
                let u = *self.entity_to_node.get(src.as_str())?;
                let v = *self.entity_to_node.get(dst.as_str())?;
                Some((u, v, label.as_str()))
            })
            .collect();
        GraphReport::build(&self.graph, &self.embedding, &resolved)
    }
}

// ============================================================================
// NodeInterner
// ============================================================================

/// Thread-safe string → u64 intern table.
pub struct NodeInterner {
    counter: AtomicU64,
    forward: RwLock<HashMap<String, u64>>,
    reverse: RwLock<HashMap<u64, String>>,
}

impl Default for NodeInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeInterner {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(1),
            forward: RwLock::new(HashMap::new()),
            reverse: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_or_intern(&self, val: &str) -> u64 {
        {
            let r = self.forward.read().unwrap();
            if let Some(&id) = r.get(val) {
                return id;
            }
        }
        let mut wf = self.forward.write().unwrap();
        if let Some(&id) = wf.get(val) {
            return id;
        }
        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        wf.insert(val.to_string(), id);
        self.reverse.write().unwrap().insert(id, val.to_string());
        id
    }
}

// ============================================================================
// Qdrant value helpers
// ============================================================================

mod qdrant_val {
    use qdrant_client::qdrant::{value::Kind, Value};

    pub fn as_str(v: &Value) -> Option<&str> {
        match v.kind.as_ref()? {
            Kind::StringValue(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(v: &Value) -> Option<f64> {
        match v.kind.as_ref()? {
            Kind::DoubleValue(d)  => Some(*d),
            Kind::IntegerValue(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_bool(v: &Value) -> Option<bool> {
        match v.kind.as_ref()? {
            Kind::BoolValue(b) => Some(*b),
            _ => None,
        }
    }
}

// ============================================================================
// Ingest result
// ============================================================================

#[derive(Debug)]
pub struct IngestResult {
    pub anomaly_score:   f32,
    pub is_anomaly:      bool,
    pub report:          Option<IncidentReport>,
    /// Regularizer status for this tick — always present after the first ingest.
    pub reg_report:      Option<RegularizationReport>,
}

// ============================================================================
// QdrantSpectralSecurityEngine
// ============================================================================

pub struct QdrantSpectralSecurityEngine {
    pub qdrant:         Qdrant,
    pub interner:       Arc<NodeInterner>,
    model:              AnomalyModel<NdArray>,
    device:             NdArrayDevice,
    threshold:          f32,
    edge_counter:       AtomicU64,
    blast_radius_depth: usize,
    pub observed_edges: RwLock<Vec<(String, String)>>,
    spectral_cfg:       JacobiConfig,
    hnsw:               RwLock<HnswFlowIndex>,
    /// Dynamic Laplacian regulariser — resolves λ₂=0 on disconnected topologies
    /// by injecting similarity-scored virtual edges and an εI diagonal shift.
    pub regularizer:    RwLock<DynamicLaplacianRegularizer>,
    /// Monotonic regularizer tick counter.
    reg_tick:           AtomicU64,
}

impl QdrantSpectralSecurityEngine {
    pub async fn new(
        url:         &str,
        model:       AnomalyModel<NdArray>,
        device:      NdArrayDevice,
        threshold:   f32,
        blast_depth: usize,
    ) -> Result<Self, EngineError> {
        let qdrant = Qdrant::from_url(url).build()?;
        let engine = Self {
            qdrant,
            interner:           Arc::new(NodeInterner::new()),
            model, device, threshold,
            edge_counter:       AtomicU64::new(10_000),
            blast_radius_depth: blast_depth,
            observed_edges:     RwLock::new(Vec::new()),
            spectral_cfg:       JacobiConfig::default(),
            hnsw:               RwLock::new(HnswFlowIndex::new()),
            // ε=0.01 prevents λ₂=0 collapse; α=0.5 scales virtual-edge weights
            regularizer:        RwLock::new(DynamicLaplacianRegularizer::new(0, 0.01, 0.5)),
            reg_tick:           AtomicU64::new(0),
        };
        engine.bootstrap(true).await?;
        Ok(engine)
    }

    async fn bootstrap(&self, reset: bool) -> Result<(), EngineError> {
        if reset && self.qdrant.collection_exists(COLLECTION_NAME).await? {
            self.qdrant.delete_collection(COLLECTION_NAME).await?;
            info!("Dropped stale collection '{COLLECTION_NAME}'.");
        }
        if !self.qdrant.collection_exists(COLLECTION_NAME).await? {
            self.qdrant
                .create_collection(
                    CreateCollectionBuilder::new(COLLECTION_NAME).vectors_config(
                        VectorParamsBuilder::new(FEATURE_DIM as u64, Distance::Cosine),
                    ),
                )
                .await?;
            for field in &["src", "dst", "cic_label"] {
                self.qdrant
                    .create_field_index(CreateFieldIndexCollectionBuilder::new(
                        COLLECTION_NAME,
                        *field,
                        FieldType::Keyword,
                    ))
                    .await?;
            }
            self.qdrant
                .create_field_index(CreateFieldIndexCollectionBuilder::new(
                    COLLECTION_NAME,
                    "is_anomaly",
                    FieldType::Bool,
                ))
                .await?;
            info!("Collection '{COLLECTION_NAME}' bootstrapped.");
        }
        Ok(())
    }

    fn rebuild_spectral(&self) -> Option<SpectralSecurityGraph> {
        let edges = self.observed_edges.read().unwrap();
        if edges.len() < 2 {
            return None;
        }
        match SpectralSecurityGraph::build(&edges, &self.spectral_cfg) {
            Ok(sg) => Some(sg),
            Err(e) => {
                warn!("Spectral rebuild: {e}");
                None
            }
        }
    }

    /// Ingest one CIC-IDS2018 flow row.  Returns an [`IngestResult`] which
    /// contains an [`IncidentReport`] iff the flow is anomalous.
    pub async fn ingest_cic(&self, row: &CicRow) -> Result<IngestResult, EngineError> {
        let (src_entity, dst_entity, etype) = row.graph_edge();
        let src_str = src_entity.to_string();
        let dst_str = dst_entity.to_string();
        let ts      = row.epoch_ts();

        self.interner.get_or_intern(&src_str);
        self.interner.get_or_intern(&dst_str);
        {
            self.observed_edges
                .write()
                .unwrap()
                .push((src_str.clone(), dst_str.clone()));
        }

        // ── Regularizer tick ─────────────────────────────────────────────────
        // Rebuild the regulariser's physical edge list from observed_edges so
        // it always reflects the current network topology.  Node ids are
        // derived from the interner to stay consistent with spectral indices.
        let reg_report: Option<RegularizationReport> = {
            let edges_snapshot = self.observed_edges.read().unwrap().clone();
            // Collect unique nodes
            let mut node_set: Vec<String> = Vec::new();
            let mut node_idx: HashMap<String, u32> = HashMap::new();
            for (s, d) in &edges_snapshot {
                for e in [s, d] {
                    if !node_idx.contains_key(e.as_str()) {
                        let idx = node_set.len() as u32;
                        node_idx.insert(e.clone(), idx);
                        node_set.push(e.clone());
                    }
                }
            }
            let n = node_set.len();
            if n >= 2 {
                let phys: Vec<(u32, u32, f32)> = edges_snapshot
                    .iter()
                    .filter_map(|(s, d)| {
                        let u = *node_idx.get(s.as_str())?;
                        let v = *node_idx.get(d.as_str())?;
                        if u != v { Some((u, v, 1.0)) } else { None }
                    })
                    .collect();
                let tick = self.reg_tick.fetch_add(1, Ordering::Relaxed);
                let mut reg = self.regularizer.write().unwrap();
                // Resize the regulariser if the node count has grown
                if reg.fiedler_vec.len() != n {
                    *reg = DynamicLaplacianRegularizer::new(n, 0.01, 0.5);
                }
                reg.update_physical_edges(phys);
                Some(reg.tick(tick))
            } else {
                None
            }
        };

        let features = embed_cic_numeric(row);
        let tensor = Tensor::<NdArray, 1>::from_data(
            TensorData::new(features.clone(), Shape::new([FEATURE_DIM])),
            &self.device,
        )
        .unsqueeze::<2>();
        let (score, _bucket_errors) = self.model.score(tensor);
        let is_anomaly = score > self.threshold;

        let similar_flows: Vec<HnswMatch> = if is_anomaly {
            self.hnsw.read().unwrap().knn(&features, HNSW_KNN)
        } else {
            Vec::new()
        };

        self.hnsw.write().unwrap().insert(HnswPoint {
            features: features.clone(),
            label:    row.label.clone(),
            src:      src_str.clone(),
            dst:      dst_str.clone(),
        });

        let spectral_graph: Option<SpectralSecurityGraph> =
            if is_anomaly { self.rebuild_spectral() } else { None };

        let report: Option<IncidentReport> = if is_anomaly {
            let severity = Severity::from_loss(score, self.threshold);
            let mitre    = classify_mitre_from_label(&row.label);

            let spectral: Option<SpectralMeta> = spectral_graph.as_ref().map(|sg| {
                let (ct, fd) = sg.pair_distances(&src_str, &dst_str);
                let blast    = sg.spectral_blast_radius(&src_str, 10);
                SpectralMeta {
                    commute_time_distance:  ct,
                    fiedler_distance:       fd,
                    algebraic_connectivity: sg.algebraic_connectivity(),
                    spectral_blast_radius:  blast,
                }
            });

            let summary = {
                let tactic = match &mitre {
                    Some(t) => format!(" — {} ({})", t.tactic_name, t.tactic_id),
                    None    => String::new(),
                };
                let hnsw_note = if !similar_flows.is_empty() {
                    format!(
                        " | HNSW nn={} d={:.3}",
                        similar_flows[0].label, similar_flows[0].distance
                    )
                } else {
                    String::new()
                };
                let spectral_note = match &spectral {
                    Some(sp) if sp.fiedler_distance > 0.3 => {
                        format!(" | ⚠ cross-cluster Fd={:.3}", sp.fiedler_distance)
                    }
                    Some(sp) if sp.algebraic_connectivity < 0.1 => {
                        format!(" | ⚠ bridge λ₁={:.4}", sp.algebraic_connectivity)
                    }
                    _ => String::new(),
                };
                format!(
                    "[{severity}] {src_str} → {dst_str} | MSE={score:.4}{tactic}{hnsw_note}{spectral_note}"
                )
            };

            let blast_radius_nodes = self
                .trace_bfs_blast_radius(src_entity.clone(), self.blast_radius_depth)
                .await?
                .into_iter()
                .collect();

            warn!(
                severity = %severity, score, label = %row.label,
                mitre = mitre.as_ref().map(|m| m.tactic_id.as_str()).unwrap_or("none"),
                hnsw_nn = similar_flows.first().map(|m| m.label.as_str()).unwrap_or("-"),
                "[ANOMALY]"
            );

            Some(IncidentReport {
                summary,
                severity,
                source:        src_str.clone(),
                destination:   dst_str.clone(),
                edge_type:     etype.as_str().to_string(),
                timestamp:     ts,
                anomaly_score: score,
                mitre_tactic:  mitre,
                blast_radius_nodes,
                spectral,
                similar_flows,
                raw_label:     row.label.clone(),
            })
        } else {
            info!(score, label = %row.label, "Within baseline");
            None
        };

        let point_id = self.edge_counter.fetch_add(1, Ordering::Relaxed);

        let (ct_dist, fd_dist, lambda1) = report
            .as_ref()
            .and_then(|r| r.spectral.as_ref())
            .map(|sp| {
                (
                    sp.commute_time_distance,
                    sp.fiedler_distance,
                    sp.algebraic_connectivity,
                )
            })
            .unwrap_or((0.0, 0.0, 0.0));

        let hnsw_nn_label = report
            .as_ref()
            .and_then(|r| r.similar_flows.first())
            .map(|m| m.label.as_str())
            .unwrap_or("none");

        let json_payload = serde_json::json!({
            "src":                    &src_str,
            "dst":                    &dst_str,
            "etype":                  etype.as_str(),
            "cic_label":              &row.label,
            "timestamp":              ts,
            "anomaly_score":          score,
            "is_anomaly":             is_anomaly,
            "severity":               report.as_ref()
                                          .map(|r| r.severity.to_string())
                                          .unwrap_or_else(|| "INFO".into()),
            "mitre_tactic_id":        report.as_ref()
                                          .and_then(|r| r.mitre_tactic.as_ref())
                                          .map(|m| m.tactic_id.as_str())
                                          .unwrap_or("none"),
            "summary":                report.as_ref()
                                          .map(|r| r.summary.clone())
                                          .unwrap_or_default(),
            "hnsw_nn_label":          hnsw_nn_label,
            "commute_time_distance":  ct_dist,
            "fiedler_distance":       fd_dist,
            "algebraic_connectivity": lambda1,
        });

        let qdrant_payload: Payload = json_payload
            .try_into()
            .map_err(|e: QdrantError| EngineError::Qdrant(e))?;

        self.qdrant
            .upsert_points(UpsertPointsBuilder::new(
                COLLECTION_NAME,
                vec![PointStruct::new(point_id, features, qdrant_payload)],
            ))
            .await?;

        Ok(IngestResult { anomaly_score: score, is_anomaly, report, reg_report })
    }

    /// BFS over Qdrant `dst` edges starting from `seed`.
    pub async fn trace_bfs_blast_radius(
        &self,
        seed:      Entity,
        max_depth: usize,
    ) -> Result<HashSet<String>, EngineError> {
        let root = seed.to_string();
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier = vec![root.clone()];
        visited.insert(root);

        for depth in 0..max_depth {
            if frontier.is_empty() {
                break;
            }
            info!(depth, n = frontier.len(), "BFS blast radius");
            let mut next: Vec<String> = Vec::new();
            for cur in &frontier {
                let filter =
                    Filter::must([Condition::matches("src", cur.clone())]);
                let resp = self
                    .qdrant
                    .scroll(
                        ScrollPointsBuilder::new(COLLECTION_NAME)
                            .filter(filter)
                            .limit(100)
                            .with_payload(true),
                    )
                    .await?;
                for pt in resp.result {
                    if let Some(v) = pt.payload.get("dst") {
                        if let Some(s) = qdrant_val::as_str(v) {
                            if visited.insert(s.to_string()) {
                                next.push(s.to_string());
                            }
                        }
                    }
                }
            }
            frontier = next;
        }
        Ok(visited)
    }

    /// Read all points back from Qdrant and compute detection metrics.
    pub async fn query_and_evaluate(&self) -> Result<TestMetrics, EngineError> {
        let mut all_points = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;

        loop {
            let mut builder = ScrollPointsBuilder::new(COLLECTION_NAME)
                .limit(500)
                .with_payload(true);
            if let Some(ref off) = offset {
                builder = builder.offset(off.clone());
            }
            let resp = self.qdrant.scroll(builder).await?;
            let fetched = resp.result.len();
            all_points.extend(resp.result);
            match resp.next_page_offset {
                Some(next) if fetched > 0 => offset = Some(next),
                _                         => break,
            }
        }

        let (mut tp, mut fp, mut tn, mut fn_) = (0u32, 0u32, 0u32, 0u32);
        let mut detections: Vec<DetectionRow> = Vec::new();

        for pt in &all_points {
            let p         = &pt.payload;
            let cic_label = p.get("cic_label")
                .and_then(|v| qdrant_val::as_str(v))
                .unwrap_or("Benign");
            let is_anomaly = p.get("is_anomaly")
                .and_then(|v| qdrant_val::as_bool(v))
                .unwrap_or(false);
            let is_attack = cic_label != "Benign";
            match (is_anomaly, is_attack) {
                (true,  true)  => tp  += 1,
                (true,  false) => fp  += 1,
                (false, false) => tn  += 1,
                (false, true)  => fn_ += 1,
            }
            detections.push(DetectionRow {
                cic_label:              cic_label.to_string(),
                detected:               is_anomaly,
                score:                  p.get("anomaly_score")
                    .and_then(|v| qdrant_val::as_f64(v))
                    .unwrap_or(0.0) as f32,
                severity:               p.get("severity")
                    .and_then(|v| qdrant_val::as_str(v))
                    .unwrap_or("INFO")
                    .to_string(),
                mitre_tactic:           p.get("mitre_tactic_id")
                    .and_then(|v| qdrant_val::as_str(v))
                    .unwrap_or("none")
                    .to_string(),
                hnsw_nn:                p.get("hnsw_nn_label")
                    .and_then(|v| qdrant_val::as_str(v))
                    .unwrap_or("-")
                    .to_string(),
                commute_time_distance:  p.get("commute_time_distance")
                    .and_then(|v| qdrant_val::as_f64(v))
                    .unwrap_or(0.0),
                fiedler_distance:       p.get("fiedler_distance")
                    .and_then(|v| qdrant_val::as_f64(v))
                    .unwrap_or(0.0),
                algebraic_connectivity: p.get("algebraic_connectivity")
                    .and_then(|v| qdrant_val::as_f64(v))
                    .unwrap_or(0.0),
                src:                    p.get("src")
                    .and_then(|v| qdrant_val::as_str(v))
                    .unwrap_or("?")
                    .to_string(),
                dst:                    p.get("dst")
                    .and_then(|v| qdrant_val::as_str(v))
                    .unwrap_or("?")
                    .to_string(),
            });
        }
        Ok(TestMetrics { tp, fp, tn, fn_, detections })
    }
}

// ============================================================================
// CSE-CIC-IDS2018 row definition
// ============================================================================

/// Represents one parsed row from the CSE-CIC-IDS2018 dataset.
#[derive(Debug, Clone)]
pub struct CicRow {
    pub src_ip:              String,
    pub dst_ip:              String,
    pub dst_port:            u32,
    pub protocol:            u8,
    pub timestamp:           String,
    pub flow_duration:       i64,
    pub tot_fwd_pkts:        u64,
    pub tot_bwd_pkts:        u64,
    pub totlen_fwd_pkts:     f64,
    pub totlen_bwd_pkts:     f64,
    pub fwd_pkt_len_max:     f64,
    pub fwd_pkt_len_min:     f64,
    pub fwd_pkt_len_mean:    f64,
    pub fwd_pkt_len_std:     f64,
    pub bwd_pkt_len_max:     f64,
    pub bwd_pkt_len_min:     f64,
    pub bwd_pkt_len_mean:    f64,
    pub bwd_pkt_len_std:     f64,
    pub flow_byts_s:         f64,
    pub flow_pkts_s:         f64,
    pub flow_iat_mean:       f64,
    pub flow_iat_std:        f64,
    pub flow_iat_max:        f64,
    pub flow_iat_min:        f64,
    pub fwd_iat_tot:         f64,
    pub fwd_iat_mean:        f64,
    pub fwd_iat_std:         f64,
    pub fwd_iat_max:         f64,
    pub fwd_iat_min:         f64,
    pub bwd_iat_tot:         f64,
    pub bwd_iat_mean:        f64,
    pub bwd_iat_std:         f64,
    pub bwd_iat_max:         f64,
    pub bwd_iat_min:         f64,
    pub fwd_psh_flags:       u8,
    pub bwd_psh_flags:       u8,
    pub fwd_urg_flags:       u8,
    pub bwd_urg_flags:       u8,
    pub fwd_header_len:      u32,
    pub bwd_header_len:      u32,
    pub fwd_pkts_s:          f64,
    pub bwd_pkts_s:          f64,
    pub pkt_len_min:         f64,
    pub pkt_len_max:         f64,
    pub pkt_len_mean:        f64,
    pub pkt_len_std:         f64,
    pub pkt_len_var:         f64,
    pub fin_flag_cnt:        u8,
    pub syn_flag_cnt:        u8,
    pub rst_flag_cnt:        u8,
    pub psh_flag_cnt:        u8,
    pub ack_flag_cnt:        u8,
    pub urg_flag_cnt:        u8,
    pub cwe_flag_count:      u8,
    pub ece_flag_cnt:        u8,
    pub down_up_ratio:       f64,
    pub pkt_size_avg:        f64,
    pub fwd_seg_size_avg:    f64,
    pub bwd_seg_size_avg:    f64,
    pub fwd_byts_b_avg:      f64,
    pub fwd_pkts_b_avg:      f64,
    pub fwd_blk_rate_avg:    f64,
    pub bwd_byts_b_avg:      f64,
    pub bwd_pkts_b_avg:      f64,
    pub bwd_blk_rate_avg:    f64,
    pub subflow_fwd_pkts:    u64,
    pub subflow_fwd_byts:    u64,
    pub subflow_bwd_pkts:    u64,
    pub subflow_bwd_byts:    u64,
    pub init_fwd_win_byts:   i64,
    pub init_bwd_win_byts:   i64,
    pub fwd_act_data_pkts:   u64,
    pub fwd_seg_size_min:    u32,
    pub active_mean:         f64,
    pub active_std:          f64,
    pub active_max:          f64,
    pub active_min:          f64,
    pub idle_mean:           f64,
    pub idle_std:            f64,
    pub idle_max:            f64,
    pub idle_min:            f64,
    pub label:               String,
}

impl CicRow {
    pub fn graph_edge(&self) -> (Entity, Entity, EdgeType) {
        let src   = Entity::IpAddress(self.src_ip.clone());
        let dst   = Entity::IpAddress(self.dst_ip.clone());
        let etype = match (self.protocol, self.dst_port) {
            (6, 22) | (6, 23) | (6, 3389) | (6, 21) => EdgeType::AuthenticatedTo,
            (6, 80) | (6, 443) | (6, 8080)           => EdgeType::CommunicatedWith,
            (17, _) if self.flow_pkts_s > 1_000.0    => EdgeType::ObservedWith,
            _                                         => EdgeType::CommunicatedWith,
        };
        (src, dst, etype)
    }

    /// Approximate Unix timestamp from the dataset's `DD/MM/YYYY HH:MM:SS` format.
    pub fn epoch_ts(&self) -> u64 {
        let parts: Vec<&str> = self.timestamp.split_whitespace().collect();
        if parts.len() != 2 {
            return 0;
        }
        let date: Vec<&str> = parts[0].split('/').collect();
        let time: Vec<&str> = parts[1].split(':').collect();
        if date.len() != 3 || time.len() != 3 {
            return 0;
        }
        let (d, m, y) = (
            date[0].parse::<u64>().unwrap_or(1),
            date[1].parse::<u64>().unwrap_or(1),
            date[2].parse::<u64>().unwrap_or(2018),
        );
        let (h, mn, s) = (
            time[0].parse::<u64>().unwrap_or(0),
            time[1].parse::<u64>().unwrap_or(0),
            time[2].parse::<u64>().unwrap_or(0),
        );
        (y - 1970) * 365 * 86400
            + (m - 1) * 30 * 86400
            + (d - 1) * 86400
            + h * 3600
            + mn * 60
            + s
    }
}

// ============================================================================
// Metrics
// ============================================================================

#[derive(Debug)]
pub struct DetectionRow {
    pub cic_label:              String,
    pub detected:               bool,
    pub score:                  f32,
    pub severity:               String,
    pub mitre_tactic:           String,
    pub hnsw_nn:                String,
    pub commute_time_distance:  f64,
    pub fiedler_distance:       f64,
    pub algebraic_connectivity: f64,
    pub src:                    String,
    pub dst:                    String,
}

#[derive(Debug)]
pub struct TestMetrics {
    pub tp:         u32,
    pub fp:         u32,
    pub tn:         u32,
    pub fn_:        u32,
    pub detections: Vec<DetectionRow>,
}

impl TestMetrics {
    pub fn precision(&self) -> f64 {
        let d = self.tp + self.fp;
        if d == 0 { 0.0 } else { self.tp as f64 / d as f64 }
    }
    pub fn recall(&self) -> f64 {
        let d = self.tp + self.fn_;
        if d == 0 { 0.0 } else { self.tp as f64 / d as f64 }
    }
    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
    pub fn accuracy(&self) -> f64 {
        let t = self.tp + self.fp + self.tn + self.fn_;
        if t == 0 { 0.0 } else { (self.tp + self.tn) as f64 / t as f64 }
    }
}

impl fmt::Display for TestMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "┌──────────────────────────────────────────────────────────────────────────────────┐")?;
        writeln!(f, "│   CSE-CIC-IDS2018 × SPECTRAL + HNSW  DETECTION METRICS                          │")?;
        writeln!(f, "├──────────────────────────────────────────────────────────────────────────────────┤")?;
        writeln!(f, "│  TP:{:>5}  FP:{:>5}  TN:{:>5}  FN:{:>5}                                         │",
            self.tp, self.fp, self.tn, self.fn_)?;
        writeln!(f, "│  Precision={:.3}  Recall={:.3}  F1={:.3}  Accuracy={:.3}                       │",
            self.precision(), self.recall(), self.f1(), self.accuracy())?;
        writeln!(f, "├──────────────────────────────────────────────────────────────────────────────────┤")?;
        writeln!(f, "│  {:<32} {:>7}  {:<8} {:<8}  {:<12}  {}",
            "CIC Label", "Score", "Det.", "Severity", "HNSW-NN", "MITRE")?;
        writeln!(f, "│  {}", "─".repeat(80))?;
        for d in &self.detections {
            let label_trunc = &d.cic_label[..d.cic_label.len().min(32)];
            let nn_trunc    = &d.hnsw_nn[..d.hnsw_nn.len().min(12)];
            writeln!(f, "│  {:<32} {:>7.4}  {:<8} {:<8}  {:<12}  {}",
                label_trunc,
                d.score,
                if d.detected { "✓ ALERT" } else { "  ok" },
                d.severity,
                nn_trunc,
                d.mitre_tactic)?;
        }
        write!(f, "└──────────────────────────────────────────────────────────────────────────────────┘")
    }
}

// ============================================================================
// Synthetic dataset
// ============================================================================

/// A representative set of synthetic CIC-IDS2018 rows covering benign
/// traffic and 13 attack categories for unit/integration testing.
pub fn ids2018_sample_dataset() -> Vec<CicRow> {
    macro_rules! row {
        ($label:expr,$src:expr,$dst:expr,$dport:expr,$proto:expr,
         $ts:expr,$fdur:expr,$tfwd:expr,$tbwd:expr,
         $bps:expr,$pps:expr,$fpm:expr,$bpm:expr,
         $syn:expr,$rst:expr,$fin:expr,$psh:expr,$ack:expr,
         $plm:expr,$pls:expr,$ifw:expr,$ibw:expr,
         $idle:expr,$active:expr,$fiat:expr,$fiatmax:expr) => {
            CicRow {
                src_ip: $src.into(), dst_ip: $dst.into(),
                dst_port: $dport, protocol: $proto,
                timestamp: $ts.into(), flow_duration: $fdur,
                tot_fwd_pkts: $tfwd, tot_bwd_pkts: $tbwd,
                totlen_fwd_pkts: $fpm * $tfwd as f64,
                totlen_bwd_pkts: $bpm * $tbwd as f64,
                fwd_pkt_len_max: $fpm * 2.0, fwd_pkt_len_min: 0.0,
                fwd_pkt_len_mean: $fpm, fwd_pkt_len_std: $fpm * 0.3,
                bwd_pkt_len_max: $bpm * 2.0, bwd_pkt_len_min: 0.0,
                bwd_pkt_len_mean: $bpm, bwd_pkt_len_std: $bpm * 0.3,
                flow_byts_s: $bps, flow_pkts_s: $pps,
                flow_iat_mean: $fiat, flow_iat_std: $fiat * 0.5,
                flow_iat_max: $fiatmax, flow_iat_min: 0.0,
                fwd_iat_tot: ($fdur as f64).max(0.0), fwd_iat_mean: $fiat,
                fwd_iat_std: $fiat * 0.4, fwd_iat_max: $fiatmax, fwd_iat_min: 0.0,
                bwd_iat_tot: 0.0, bwd_iat_mean: 0.0, bwd_iat_std: 0.0,
                bwd_iat_max: 0.0, bwd_iat_min: 0.0,
                fwd_psh_flags: 0, bwd_psh_flags: 0,
                fwd_urg_flags: 0, bwd_urg_flags: 0,
                fwd_header_len: 20, bwd_header_len: 20,
                fwd_pkts_s: $pps / 2.0, bwd_pkts_s: $pps / 2.0,
                pkt_len_min: 0.0, pkt_len_max: $plm + $pls * 3.0,
                pkt_len_mean: $plm, pkt_len_std: $pls, pkt_len_var: $pls * $pls,
                fin_flag_cnt: $fin, syn_flag_cnt: $syn,
                rst_flag_cnt: $rst, psh_flag_cnt: $psh,
                ack_flag_cnt: $ack, urg_flag_cnt: 0,
                cwe_flag_count: 0, ece_flag_cnt: 0,
                down_up_ratio: if $tbwd > 0 { $tbwd as f64 / $tfwd as f64 } else { 0.0 },
                pkt_size_avg: $plm,
                fwd_seg_size_avg: $fpm, bwd_seg_size_avg: $bpm,
                fwd_byts_b_avg: 0.0, fwd_pkts_b_avg: 0.0, fwd_blk_rate_avg: 0.0,
                bwd_byts_b_avg: 0.0, bwd_pkts_b_avg: 0.0, bwd_blk_rate_avg: 0.0,
                subflow_fwd_pkts: $tfwd,
                subflow_fwd_byts: ($fpm * $tfwd as f64) as u64,
                subflow_bwd_pkts: $tbwd,
                subflow_bwd_byts: ($bpm * $tbwd as f64) as u64,
                init_fwd_win_byts: $ifw, init_bwd_win_byts: $ibw,
                fwd_act_data_pkts: $tfwd, fwd_seg_size_min: 20,
                active_mean: $active, active_std: 0.0,
                active_max: $active, active_min: $active,
                idle_mean: $idle, idle_std: 0.0,
                idle_max: $idle, idle_min: $idle,
                label: $label.into(),
            }
        };
    }

    vec![
        // ── Benign baseline ──────────────────────────────────────────────────
        row!("Benign","172.31.69.11","172.31.69.20",3389,6,"01/03/2018 09:56:59",
             4_046_191_i64,14,7,439.4,5.19,99.0,56.0,0,1,0,1,0,80.8,161.5,8192_i64,62614_i64,0.0,0.0,202309.0,957090.0),
        row!("Benign","172.31.69.12","172.31.69.2",53,17,"01/03/2018 09:57:00",
             303_i64,1,1,356435.6,6600.7,46.0,62.0,0,0,0,0,0,51.3,9.2,-1_i64,-1_i64,0.0,0.0,303.0,303.0),
        row!("Benign","172.31.69.30","54.239.28.85",443,6,"01/03/2018 10:01:00",
             120_000_000_i64,45,38,18_432.0,0.69,312.0,820.0,1,0,1,12,40,530.0,480.0,65535_i64,65535_i64,0.0,0.0,2_666_667.0,15_000_000.0),
        row!("Benign","172.31.69.50","172.31.69.2",53,17,"01/03/2018 08:00:00",
             400_i64,1,1,300_000.0,5000.0,48.0,60.0,0,0,0,0,0,54.0,8.0,-1_i64,-1_i64,0.0,0.0,400.0,400.0),
        row!("Benign","172.31.69.51","52.84.100.12",443,6,"01/03/2018 08:05:00",
             60_000_000_i64,30,25,9_600.0,0.92,280.0,640.0,1,0,1,8,28,430.0,390.0,65535_i64,65535_i64,0.0,0.0,2_000_000.0,12_000_000.0),
        row!("Benign","172.31.69.52","172.31.69.10",80,6,"01/03/2018 08:10:00",
             3_000_000_i64,8,6,6_400.0,4.67,350.0,980.0,1,0,1,2,6,620.0,450.0,65535_i64,65535_i64,0.0,0.0,428_571.0,2_000_000.0),
        // ── Attacks ──────────────────────────────────────────────────────────
        row!("Infilteration","172.31.69.95","172.31.69.20",3389,6,"01/03/2018 09:57:00",
             1_402_669_i64,8,7,1945.6,10.69,143.5,225.9,0,0,0,1,0,181.9,319.4,8192_i64,62852_i64,0.0,0.0,200381.0,1_073_843.0),
        row!("Bot","172.31.69.88","52.70.12.33",443,6,"28/02/2018 14:22:10",
             600_000_000_i64,120,120,3_200.0,0.4,128.0,128.0,0,0,0,120,120,128.0,0.0,65535_i64,65535_i64,0.0,0.0,5_000_000.0,5_000_100.0),
        row!("DoS attacks-Hulk","172.31.69.200","172.31.69.10",80,6,"20/02/2018 09:10:00",
             60_000_000_i64,9_000,0,72_000_000.0,150_000.0,480.0,0.0,1,0,0,1,1,480.0,0.0,8192_i64,-1_i64,0.0,0.0,400.0,1_200.0),
        row!("DoS attacks-GoldenEye","172.31.69.201","172.31.69.10",80,6,"20/02/2018 11:30:00",
             120_000_000_i64,1_800,400,6_400_000.0,18_333.0,1_400.0,200.0,1,0,0,1,1,1_100.0,600.0,512_i64,512_i64,0.0,0.0,3_333.0,20_000.0),
        row!("DDoS attacks-LOIC-HTTP","172.31.69.210","172.31.69.10",80,17,"28/02/2018 10:05:00",
             10_000_000_i64,500_000,0,400_000_000.0,50_000_000.0,64.0,0.0,0,0,0,0,0,64.0,0.0,-1_i64,-1_i64,0.0,0.0,10.0,50.0),
        row!("Brute Force -Web","172.31.69.180","172.31.69.10",8080,6,"22/02/2018 10:20:00",
             5_000_000_i64,6_000,6_000,3_200_000.0,2_400.0,400.0,360.0,1,1,1,1,1,380.0,60.0,512_i64,16384_i64,0.0,0.0,833.0,2_000.0),
        row!("SQL Injection","172.31.69.181","172.31.69.10",80,6,"22/02/2018 11:45:00",
             200_000_i64,3,2,86_000.0,25.0,4_096.0,512.0,1,0,1,1,1,2_730.0,1_820.0,65535_i64,65535_i64,0.0,0.0,100_000.0,200_000.0),
        row!("XSS","172.31.69.182","172.31.69.10",80,6,"22/02/2018 12:10:00",
             180_000_i64,4,3,112_000.0,38.9,2_048.0,400.0,1,0,1,1,1,1_366.0,900.0,65535_i64,65535_i64,0.0,0.0,60_000.0,120_000.0),
        row!("Brute Force -XSS","172.31.69.190","172.31.69.15",22,6,"16/02/2018 09:00:00",
             300_000_000_i64,180_000,180_000,5_760_000.0,1_200.0,40.0,40.0,1,1,0,0,0,40.0,0.0,512_i64,-1_i64,0.0,0.0,833.0,1_000.0),
        row!("FTP-BruteForce","172.31.69.191","172.31.69.14",21,6,"16/02/2018 10:30:00",
             200_000_000_i64,90_000,90_000,720_000.0,900.0,40.0,40.0,1,1,0,0,0,40.0,0.0,512_i64,-1_i64,0.0,0.0,555.0,800.0),
        row!("Heartbleed","172.31.69.99","172.31.69.16",443,6,"16/02/2018 14:05:00",
             800_000_i64,5,3,2_250.0,10.0,60.0,65536.0,1,0,1,1,1,16_400.0,31_200.0,65535_i64,65535_i64,0.0,0.0,200_000.0,400_000.0),
        row!("DoS attacks-SlowHTTPTest","172.31.69.202","172.31.69.10",80,6,"20/02/2018 14:00:00",
             200_000_000_000_i64,4,1,0.5,0.00002,200.0,100.0,1,0,0,1,1,175.0,70.0,65535_i64,65535_i64,86_400_000_000.0,200_000.0,66_666_666_666.0,200_000_000_000.0),
        row!("DoS attacks-Slowloris","172.31.69.203","172.31.69.10",80,6,"20/02/2018 15:00:00",
             300_000_000_000_i64,3,1,0.3,0.00001,180.0,80.0,1,0,0,0,1,150.0,65.0,65535_i64,65535_i64,120_000_000_000.0,180_000.0,100_000_000_000.0,300_000_000_000.0),
        row!("DDoS attacks-LOIC-UDP","172.31.69.211","172.31.69.10",80,17,"28/02/2018 11:00:00",
             5_000_000_i64,800_000,0,819_200_000.0,160_000_000.0,64.0,0.0,0,0,0,0,0,64.0,0.0,-1_i64,-1_i64,0.0,0.0,3.0,10.0),
    ]
}

// ============================================================================
// main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sec_net_engine=info".into()),
        )
        .init();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  CIC-IDS2018 × Spectral + HNSW Security Engine  v2  (corrected)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let device  = NdArrayDevice::Cpu;
    let dataset = ids2018_sample_dataset();

    // ── Phase 0: Pre-train on benign baseline ─────────────────────────────────
    println!("━━━  Phase 0: Pre-training autoencoder on benign flows  ━━━\n");
    let benign_rows: Vec<&CicRow> =
        dataset.iter().filter(|r| r.label == "Benign").collect();
    println!("  Benign baseline: {} flows", benign_rows.len());
    let (trained_model, threshold) =
        pretrain_on_benign(&benign_rows, &device, 200, 1e-3);
    println!("  ✓  Threshold = {threshold:.6}\n");

    // ── Phase 0b: Standalone spectral topology ────────────────────────────────
    println!("━━━  Phase 0b: Pre-ingest spectral topology  ━━━\n");
    let edges: Vec<(String, String)> = dataset
        .iter()
        .map(|r| (format!("Ip:{}", r.src_ip), format!("Ip:{}", r.dst_ip)))
        .collect();
    match SpectralSecurityGraph::build(&edges, &JacobiConfig::default()) {
        Ok(sg) => {
            let lam = sg.algebraic_connectivity();
            println!("  λ₁ = {lam:.6}");
            if lam < 1e-10 {
                println!("  ℹ  Isolated-edge topology — λ₁=0 is topologically correct.");
                println!("     CT distances on direct edges are still valid (expect CT≈1.0).\n");
            }
            let pairs = [
                ("Ip:172.31.69.200", "Ip:172.31.69.10", "DoS-Hulk → victim"),
                ("Ip:172.31.69.88",  "Ip:52.70.12.33",  "Bot → C2"),
                ("Ip:172.31.69.11",  "Ip:172.31.69.20", "Benign RDP"),
            ];
            println!("  {:<40} {:>14}  {:>12}", "Pair", "Commute-Time", "Fiedler");
            println!("  {}", "─".repeat(68));
            for (s, d, lbl) in &pairs {
                let (ct, fd) = sg.pair_distances(s, d);
                println!("  {:<40} {:>14.6}  {:>12.6}", lbl, ct, fd);
            }
        }
        Err(e) => println!("  Spectral skipped: {e}"),
    }
    println!();

    // ── Connect Qdrant ────────────────────────────────────────────────────────
    let qdrant_url = "http://localhost:6334";
    let engine = match QdrantSpectralSecurityEngine::new(
        qdrant_url,
        trained_model,
        device,
        threshold,
        2,
    )
    .await
    {
        Ok(e) => e,
        Err(e) => {
            error!(error = %e, "Cannot reach Qdrant at {qdrant_url}");
            error!("Start with: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant");
            return Ok(());
        }
    };

    // ── Phase 1: Ingest ───────────────────────────────────────────────────────
    println!("━━━  Phase 1: Ingesting {} flows  ━━━\n", dataset.len());
    let mut n_reports = 0usize;
    for row in &dataset {
        let res = engine.ingest_cic(row).await?;
        let marker = if res.is_anomaly { "⚠ ANOMALY" } else { "  ok     " };
        // Show regularizer λ₂ status inline
        let reg_tag = match &res.reg_report {
            Some(r) if r.regularizer_active =>
                format!("  [REG active λ₂_raw={:.4} → {:.4} virt={}]",
                    r.lambda2_raw, r.lambda2_regularized, r.virtual_edges_injected),
            Some(r) =>
                format!("  [λ₂={:.4}]", r.lambda2_regularized),
            None => String::new(),
        };
        println!("  [{marker}]  mse={:.5}  label={:<35}  {}→{}{}",
            res.anomaly_score, row.label, row.src_ip, row.dst_ip, reg_tag);
        if let Some(ref rep) = res.report {
            n_reports += 1;
            println!("\n{rep}\n");
        }
    }
    println!("\n  IncidentReports generated: {n_reports} / {}", dataset.len());

    // ── Phase 2: Evaluation ───────────────────────────────────────────────────
    println!("\n━━━  Phase 2: Detection evaluation (Qdrant read-back)  ━━━\n");
    let metrics = engine.query_and_evaluate().await?;
    println!("{metrics}");

    // ── Phase 3: Per-class breakdown ──────────────────────────────────────────
    println!("\n━━━  Phase 3: Per-attack-class detection rate  ━━━\n");
    let mut class_map: HashMap<String, (u32, u32)> = HashMap::new();
    for d in &metrics.detections {
        let e = class_map.entry(d.cic_label.clone()).or_insert((0, 0));
        e.0 += 1;
        if d.detected {
            e.1 += 1;
        }
    }
    let mut classes: Vec<(String, (u32, u32))> = class_map.into_iter().collect();
    classes.sort_by(|a, b| a.0.cmp(&b.0));
    println!(
        "  {:<35} {:>6}  {:>8}  {:>10}",
        "Label", "Total", "Detected", "Rate"
    );
    println!("  {}", "─".repeat(65));
    for (label, (total, detected)) in &classes {
        let rate = if *total == 0 {
            0.0
        } else {
            *detected as f64 / *total as f64 * 100.0
        };
        println!(
            "  {:<35} {:>6}  {:>8}  {:>9.1}%",
            label, total, detected, rate
        );
    }

    // ── Phase 4: Final spectral summary ──────────────────────────────────────
    println!("\n━━━  Phase 4: Final spectral topology report  ━━━\n");
    let all_edges = engine.observed_edges.read().unwrap().clone();
    if let Ok(final_sg) =
        SpectralSecurityGraph::build(&all_edges, &JacobiConfig::default())
    {
        let report_pairs: Vec<(String, String, String)> = vec![
            ("Ip:172.31.69.200".into(), "Ip:172.31.69.10".into(), "hulk→victim".into()),
            ("Ip:172.31.69.210".into(), "Ip:172.31.69.10".into(), "loic→victim".into()),
            ("Ip:172.31.69.88".into(),  "Ip:52.70.12.33".into(),  "bot→c2".into()),
        ];
        match final_sg.report(&report_pairs) {
            Ok(r)  => r.print(),
            Err(e) => println!("  Report error: {e}"),
        }
        println!("  λ₁ final: {:.6}", final_sg.algebraic_connectivity());
    }

    // ── Phase 5: Dynamic Laplacian Regularizer standalone demo ───────────────
    println!("\n━━━  Phase 5: Dynamic Laplacian Regularizer — disconnection drill  ━━━\n");
    {
        use laplacian_regularizer::{DynamicLaplacianRegularizer, NodeMeta};

        // 6-node graph: 0-1-2 connected, 3-4-5 isolated (simulates network split)
        let mut reg = DynamicLaplacianRegularizer::new(6, 0.01, 0.5);

        // Register infrastructure metadata so similarity scoring has signal
        let subnet_a: u32 = 0xC0A8_0100; // 192.168.1.0
        let subnet_b: u32 = 0xC0A8_0200; // 192.168.2.0
        for id in 0u32..3 {
            reg.register_node(id, NodeMeta {
                subnet_prefix: subnet_a,
                prefix_len:    24,
                gateway:       Some(100),
                as_path:       vec![64512, 64513],
                infra_parent:  Some(0),
            });
        }
        for id in 3u32..6 {
            reg.register_node(id, NodeMeta {
                subnet_prefix: subnet_b,
                prefix_len:    24,
                gateway:       Some(101),
                as_path:       vec![64514, 64513],
                infra_parent:  Some(1),
            });
        }

        // Tick 0 — only intra-cluster edges (two disconnected components)
        reg.update_physical_edges(vec![
            (0, 1, 1.0), (1, 2, 1.0),
            (3, 4, 1.0), (4, 5, 1.0),
        ]);
        let r0 = reg.tick(0);
        println!("  Tick 0 (split)   λ₂_raw={:.4}  λ₂_reg={:.4}  virt_injected={}  active={}",
            r0.lambda2_raw, r0.lambda2_regularized, r0.virtual_edges_injected, r0.regularizer_active);

        // Show injected virtual edges
        if !reg.virtual_edges().is_empty() {
            println!("  Virtual edges injected:");
            for ve in reg.virtual_edges().iter().take(5) {
                let reason = match &ve.reason {
                    VirtualEdgeReason::CommonGateway { gateway_id }       =>
                        format!("CommonGateway({})", gateway_id),
                    VirtualEdgeReason::SharedSubnet { prefix_len }        =>
                        format!("SharedSubnet(/{prefix_len})"),
                    VirtualEdgeReason::AsPathOverlap { overlap_score }    =>
                        format!("AsPathOverlap({:.2})", overlap_score),
                    VirtualEdgeReason::InfrastructureParent { parent_node } =>
                        format!("InfraParent({})", parent_node),
                };
                println!("    {}→{}  w={:.3}  reason={}", ve.src, ve.dst, ve.weight, reason);
            }
        }

        // Tick 1 — add cross-cluster bridge (connectivity recovers)
        reg.update_physical_edges(vec![
            (0, 1, 1.0), (1, 2, 1.0),
            (3, 4, 1.0), (4, 5, 1.0),
            (2, 3, 0.5),   // ← new physical bridge
        ]);
        let r1 = reg.tick(1);
        println!("\n  Tick 1 (bridge)  λ₂_raw={:.4}  λ₂_reg={:.4}  virt_injected={}  active={}",
            r1.lambda2_raw, r1.lambda2_regularized, r1.virtual_edges_injected, r1.regularizer_active);

        // Propagation gradients for node 2 (the bridge node)
        let grads = reg.propagation_gradient(2);
        println!("\n  Propagation gradients from node 2 (bridge):");
        for (nbr, g) in &grads {
            println!("    → node {nbr}  grad={g:+.5}");
        }
        println!("  (positive = flow away from node 2; negative = flow toward)");

        // Simulate 10 more ticks of full connectivity — watch virtual edges decay
        for tick in 2u64..12 {
            let r = reg.tick(tick);
            if tick == 11 {
                println!("\n  Tick 11 (healed) λ₂_raw={:.4}  λ₂_reg={:.4}  virt_remaining={}",
                    r.lambda2_raw, r.lambda2_regularized, reg.virtual_edges().len());
            }
        }
        println!("\n  λ₂ trend (last {} ticks): {:?}",
            reg.lambda2_trend().len(),
            reg.lambda2_trend().iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>());
    }

    println!("\n━━━  Complete  ━━━");
    Ok(())
}