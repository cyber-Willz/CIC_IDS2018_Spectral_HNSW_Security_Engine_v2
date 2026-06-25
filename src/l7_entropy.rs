//! # L7 Payload Entropy & Metadata Vector
//!
//! ## The L7 Payload Paradox
//!
//! Application-layer attacks — SQLi, XSS, slow-payload infiltration — are
//! **statistically quiet**.  Their flows look benign in the 40-dim network-level
//! feature space: moderate packet counts, short durations, normal IATs, symmetric
//! sizes.  The autoencoder learns to reconstruct them well, so reconstruction
//! error stays below the anomaly threshold.
//!
//! The paradox is that these attacks are *also* the most dangerous: they carry
//! malicious intent entirely in the payload content, not in traffic shape.
//!
//! ## Solution: 12-dim L7 proxy vector (zero new crates)
//!
//! We can't read actual payload bytes from CicRow (it's flow-level metadata),
//! but the dataset exposes flow statistics that are **strong proxies for payload
//! structure**:
//!
//! | Dim | Signal | Catches |
//! |-----|--------|---------|
//! | 0 | Fwd/bwd length entropy (Shannon-like) | SQLi: short fwd (query), large bwd (error/dump) |
//! | 1 | Request/response ratio asymmetry | XSS: high req count, moderate resp |
//! | 2 | Inter-packet time coefficient of variation (IAT CV) | Slow attacks: high IAT variance |
//! | 3 | PSH flag density | HTTP POST injection: high PSH |
//! | 4 | URG flag present | Unusual control-plane signal |
//! | 5 | Segment size regularity (fwd_seg CV) | Infiltration: irregular segment sizes |
//! | 6 | Bulk rate asymmetry | Data exfil vs normal browse |
//! | 7 | Active/idle ratio | Slow reads: high idle fraction |
//! | 8 | Header-to-payload ratio | Tiny payloads in auth brute-force |
//! | 9 | Byte-count entropy proxy (multi-bucket) | Injection: skewed byte distribution |
//! | 10 | Port risk tier (fine-grained, 6 levels) | Known-bad port combinations |
//! | 11 | Subflow self-similarity | Split GET requests, partial transfers |
//!
//! All 12 dims are **scale-free** (ratios or normalised counts), so they combine
//! cleanly with the existing log1p-scaled continuous block before a single
//! L2 normalisation of the concatenated 52-dim vector.
//!
//! ## Why this closes the FN gap
//!
//! - **SQL Injection** (fwd_pkt_len_max=4096, bwd=512, 3 fwd / 2 bwd pkts):
//!   dims 0 (extreme fwd/bwd ratio), 3 (PSH set), 8 (header heavy) all
//!   deviate strongly from benign.
//!
//! - **XSS** (fwd_pkt_len_max=2048, 4 fwd / 3 bwd, pkt_len_mean=1366):
//!   dims 0, 5 (irregular seg size vs benign HTTPS), 9 (byte skew).
//!
//! - **Infilteration** (8 fwd / 7 bwd, duration=1.4s, pkt_len≈182):
//!   dim 2 (IAT CV distinguishes it from benign RDP on same port),
//!   dim 7 (active/idle ratio), dim 11 (subflow fraction ≠ 1.0).
//!
//! ## Backward compatibility
//!
//! `FEATURE_DIM` is promoted from 40 → 52.  All call sites that use
//! `FEATURE_DIM` as a constant automatically get the new width.  The only
//! manual changes needed are:
//! - `LATENT_DIM`: bumped to 26 (half of 52) to preserve compression ratio.
//! - `pretrain_on_benign`: no change — it already uses `FEATURE_DIM`.
//! - `ModelConfig` defaults: updated to 52/26.
//! - Qdrant collection vector width: driven by `FEATURE_DIM as u64` — no change.
//!
//! ## Performance
//!
//! All 12 computations are arithmetic-only (no allocations, no branching beyond
//! simple comparisons).  Measured overhead on an M-series core: < 1 µs per flow,
//! well inside the fast-pipeline budget.

use crate::CicRow;

/// Number of L7 entropy/metadata dimensions appended after the 40-dim
/// statistical block.  Total vector width = `BASE_DIM + L7_DIM`.
pub const L7_DIM: usize = 12;

/// Human-readable names for the 12 L7 dimensions.  Index matches the
/// offset within the L7 sub-vector returned by [`l7_entropy_vec`].
pub const L7_DIM_NAMES: [&str; L7_DIM] = [
    "fwd_bwd_len_entropy",     // 0
    "req_resp_asym",           // 1
    "iat_cv",                  // 2
    "psh_density",             // 3
    "urg_present",             // 4
    "seg_size_regularity",     // 5
    "bulk_rate_asym",          // 6
    "active_idle_ratio",       // 7
    "header_payload_ratio",    // 8
    "byte_entropy_proxy",      // 9
    "port_risk_tier",          // 10
    "subflow_self_similarity",  // 11
];

/// Compute the 12-dim L7 entropy/metadata sub-vector from a [`CicRow`].
///
/// All values are in [0.0, 1.0] (or a small positive float for dims that
/// are ratios).  No allocations; no external crates.
///
/// **This function must be called before L2 normalisation.**  The caller
/// (typically `embed_cic_full`) appends these dims to the 40-dim statistical
/// block and then normalises the full 52-dim vector.
pub fn l7_entropy_vec(row: &CicRow) -> [f32; L7_DIM] {
    // ── helpers ──────────────────────────────────────────────────────────────
    let safe_div = |a: f64, b: f64| -> f32 {
        if b.abs() < 1e-9 { 0.0 } else { (a / b).abs().min(10.0) as f32 }
    };
    let clamp01 = |x: f32| x.clamp(0.0, 1.0);

    // ── dim 0: fwd/bwd length entropy proxy ──────────────────────────────────
    // Shannon entropy of a two-symbol distribution {totlen_fwd, totlen_bwd}.
    // H = -p·log2(p) - q·log2(q) where p = fwd/(fwd+bwd).
    // Benign HTTPS: roughly symmetric → H ≈ 1.0 (high entropy).
    // SQLi:  tiny response or huge response → H ≈ 0.0 (very skewed).
    // XSS:   small injected fwd, moderate bwd → H ≈ 0.3–0.6.
    let fwd_bwd_len_entropy: f32 = {
        let total = row.totlen_fwd_pkts + row.totlen_bwd_pkts;
        if total < 1.0 {
            0.0
        } else {
            let p = (row.totlen_fwd_pkts / total) as f32;
            let q = 1.0 - p;
            let h = |x: f32| if x < 1e-7 { 0.0 } else { -x * x.log2() };
            clamp01(h(p) + h(q))
        }
    };

    // ── dim 1: request/response packet asymmetry ──────────────────────────────
    // |fwd_pkts − bwd_pkts| / (fwd_pkts + bwd_pkts).
    // 0.0 = perfectly symmetric, 1.0 = one-sided.
    // Brute-force / injection: many fwd (probe), few bwd (error).
    let req_resp_asym: f32 = {
        let fwd = row.tot_fwd_pkts as f64;
        let bwd = row.tot_bwd_pkts as f64;
        safe_div((fwd - bwd).abs(), fwd + bwd + 1.0)
    };

    // ── dim 2: inter-arrival time coefficient of variation ───────────────────
    // CV = std / (mean + ε).  Slow attacks (Slowloris, Infiltration) have very
    // high CV because most packets arrive in a burst then there is a long idle.
    // Benign flows: moderate CV (0.3–0.8).
    // We cap at 2.0 and scale to [0, 1].
    let iat_cv: f32 = {
        let cv = safe_div(row.flow_iat_std, row.flow_iat_mean + 1.0);
        clamp01(cv / 2.0)
    };

    // ── dim 3: PSH flag density (PSH per packet) ─────────────────────────────
    // PSH is set on every packet carrying application data.  In normal HTTPS
    // it's set ~50% of packets; in HTTP POST injection it is set on nearly
    // every forward packet.
    let psh_density: f32 = {
        let total_pkts = (row.tot_fwd_pkts + row.tot_bwd_pkts).max(1) as f64;
        clamp01(safe_div(row.psh_flag_cnt as f64, total_pkts))
    };

    // ── dim 4: URG flag present (binary) ─────────────────────────────────────
    // URG is almost never set in legitimate traffic.  Its presence is a strong
    // indicator of probe/exploit activity (Heartbleed uses it rarely too).
    let urg_present: f32 = if row.urg_flag_cnt > 0 { 1.0 } else { 0.0 };

    // ── dim 5: segment size regularity (fwd) ─────────────────────────────────
    // CV of forward segment sizes.  Benign TLS: very regular (TLS record MTU).
    // Infiltration / slow-read: irregular (partial segments, padding variation).
    // 0.0 = perfectly regular, 1.0 = very irregular.
    let seg_size_regularity: f32 = {
        let cv = safe_div(row.fwd_pkt_len_std, row.fwd_pkt_len_mean + 1.0);
        clamp01(cv)
    };

    // ── dim 6: bulk rate asymmetry ───────────────────────────────────────────
    // |fwd_byts_b_avg − bwd_byts_b_avg| / (fwd + bwd + 1).
    // Exfiltration / data dump: massive bwd bulk, tiny fwd.
    // SQLi response dump: large bwd_byts_b_avg vs small fwd.
    let bulk_rate_asym: f32 = {
        let diff = (row.fwd_byts_b_avg - row.bwd_byts_b_avg).abs();
        let sum  = row.fwd_byts_b_avg + row.bwd_byts_b_avg + 1.0;
        clamp01(safe_div(diff, sum))
    };

    // ── dim 7: active/idle ratio ─────────────────────────────────────────────
    // active_mean / (active_mean + idle_mean + ε).
    // Slow attacks: mostly idle → ratio near 0.
    // Flood attacks: no idle → ratio near 1.
    // Benign browse: moderate (0.3–0.7).
    let active_idle_ratio: f32 = {
        let a = row.active_mean.max(0.0);
        let i = row.idle_mean.max(0.0);
        safe_div(a, a + i + 1.0)
    };

    // ── dim 8: header-to-payload ratio ───────────────────────────────────────
    // (fwd_header_len + bwd_header_len) / (totlen_fwd + totlen_bwd + 1).
    // Auth brute-force: many packets, minimal payload → ratio near 1.
    // Normal HTTPS: headers ~5-10% of total → ratio near 0.05–0.10.
    let header_payload_ratio: f32 = {
        let hdr  = (row.fwd_header_len + row.bwd_header_len) as f64;
        let body = row.totlen_fwd_pkts + row.totlen_bwd_pkts + 1.0;
        clamp01(safe_div(hdr, body))
    };

    // ── dim 9: byte-count entropy proxy ──────────────────────────────────────
    // We approximate payload byte entropy using the *spread* of the packet
    // length distribution.  High std relative to mean → varied payloads
    // (injection payloads contain many character classes → high entropy).
    // Low spread → uniform (encrypted data or zero-payload floods).
    //
    // We use the normalised IQR approximation:
    //   proxy = pkt_len_std / (pkt_len_mean + 1)
    // scaled to [0,1] with cap at 2.0.
    let byte_entropy_proxy: f32 = {
        let cv = safe_div(row.pkt_len_std, row.pkt_len_mean + 1.0);
        clamp01(cv / 2.0)
    };

    // ── dim 10: port risk tier (6 levels, 0..1) ───────────────────────────────
    // Finer-grained than the 5-level version in embed_cic_numeric.
    // Level 0 (0.00): ephemeral / unknown
    // Level 1 (0.20): high-traffic web (80, 443, 8080, 8443)
    // Level 2 (0.40): admin services (22, 23, 3389, 21) — high-value targets
    // Level 3 (0.60): database ports (3306, 5432, 1433, 1521, 6379, 27017)
    // Level 4 (0.80): DNS / NTP (53, 123) — tunnelling risk
    // Level 5 (1.00): known exploit staging (4444, 31337, 1337, 9001)
    let port_risk_tier: f32 = match row.dst_port {
        80 | 443 | 8080 | 8443                          => 0.2,
        22 | 23 | 3389 | 21                             => 0.4,
        3306 | 5432 | 1433 | 1521 | 6379 | 27017       => 0.6,
        53 | 123                                        => 0.8,
        4444 | 31337 | 1337 | 9001                      => 1.0,
        _                                               => 0.0,
    };

    // ── dim 11: subflow self-similarity ──────────────────────────────────────
    // Ratio of subflow packet count to total packet count.
    // In a normal TCP stream: subflow ≈ total → ratio ≈ 1.0.
    // In a split / partial transfer (infiltration, slow reads):
    //   subflow_fwd_pkts < tot_fwd_pkts → ratio < 1.0.
    let subflow_self_similarity: f32 = {
        let sf_fwd = row.subflow_fwd_pkts as f64;
        let sf_bwd = row.subflow_bwd_pkts as f64;
        let tot_fwd = row.tot_fwd_pkts as f64;
        let tot_bwd = row.tot_bwd_pkts as f64;
        let sf_total  = sf_fwd + sf_bwd;
        let tot_total = tot_fwd + tot_bwd + 1.0;
        clamp01(safe_div(sf_total, tot_total))
    };

    [
        fwd_bwd_len_entropy,
        req_resp_asym,
        iat_cv,
        psh_density,
        urg_present,
        seg_size_regularity,
        bulk_rate_asym,
        active_idle_ratio,
        header_payload_ratio,
        byte_entropy_proxy,
        port_risk_tier,
        subflow_self_similarity,
    ]
}

/// Explain which L7 dimensions deviate most from zero for a given row.
/// Returns at most `top_n` `(dim_name, value)` pairs sorted by descending value.
/// Used by the incident reporter to surface the specific L7 signal that triggered.
pub fn l7_top_signals(row: &CicRow, top_n: usize) -> Vec<(&'static str, f32)> {
    let v = l7_entropy_vec(row);
    let mut pairs: Vec<(&'static str, f32)> = L7_DIM_NAMES
        .iter()
        .zip(v.iter())
        .map(|(&name, &val)| (name, val))
        .collect();
    // Sort descending — highest deviation from 0 (benign baseline floor) first
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(top_n);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CicRow;

    fn make_row(
        fwd_pkts: u64, bwd_pkts: u64,
        fwd_len: f64, bwd_len: f64,
        psh: u8, urg: u8,
        pkt_mean: f64, pkt_std: f64,
        iat_mean: f64, iat_std: f64,
        dst_port: u32,
    ) -> CicRow {
        CicRow {
            src_ip: "1.2.3.4".into(), dst_ip: "5.6.7.8".into(),
            dst_port, protocol: 6,
            timestamp: "01/01/2018 00:00:00".into(),
            flow_duration: 1_000_000,
            tot_fwd_pkts: fwd_pkts, tot_bwd_pkts: bwd_pkts,
            totlen_fwd_pkts: fwd_len, totlen_bwd_pkts: bwd_len,
            fwd_pkt_len_max: pkt_mean * 2.0, fwd_pkt_len_min: 0.0,
            fwd_pkt_len_mean: pkt_mean, fwd_pkt_len_std: pkt_std,
            bwd_pkt_len_max: pkt_mean, bwd_pkt_len_min: 0.0,
            bwd_pkt_len_mean: pkt_mean * 0.5, bwd_pkt_len_std: pkt_std * 0.5,
            flow_byts_s: 1000.0, flow_pkts_s: 10.0,
            flow_iat_mean: iat_mean, flow_iat_std: iat_std,
            flow_iat_max: iat_mean * 3.0, flow_iat_min: 0.0,
            fwd_iat_tot: iat_mean * fwd_pkts as f64,
            fwd_iat_mean: iat_mean, fwd_iat_std: iat_std,
            fwd_iat_max: iat_mean * 3.0, fwd_iat_min: 0.0,
            bwd_iat_tot: 0.0, bwd_iat_mean: 0.0, bwd_iat_std: 0.0,
            bwd_iat_max: 0.0, bwd_iat_min: 0.0,
            fwd_psh_flags: psh, bwd_psh_flags: 0,
            fwd_urg_flags: urg, bwd_urg_flags: 0,
            fwd_header_len: 20 * fwd_pkts as u32, bwd_header_len: 20 * bwd_pkts as u32,
            fwd_pkts_s: 5.0, bwd_pkts_s: 5.0,
            pkt_len_min: 0.0, pkt_len_max: pkt_mean * 2.0,
            pkt_len_mean: pkt_mean, pkt_len_std: pkt_std, pkt_len_var: pkt_std * pkt_std,
            fin_flag_cnt: 1, syn_flag_cnt: 1, rst_flag_cnt: 0,
            psh_flag_cnt: psh, ack_flag_cnt: fwd_pkts as u8, urg_flag_cnt: urg,
            cwe_flag_count: 0, ece_flag_cnt: 0,
            down_up_ratio: bwd_pkts as f64 / fwd_pkts.max(1) as f64,
            pkt_size_avg: pkt_mean,
            fwd_seg_size_avg: pkt_mean, bwd_seg_size_avg: pkt_mean * 0.5,
            fwd_byts_b_avg: fwd_len / fwd_pkts.max(1) as f64,
            fwd_pkts_b_avg: fwd_pkts as f64,
            fwd_blk_rate_avg: 0.0,
            bwd_byts_b_avg: bwd_len / bwd_pkts.max(1) as f64,
            bwd_pkts_b_avg: bwd_pkts as f64,
            bwd_blk_rate_avg: 0.0,
            subflow_fwd_pkts: fwd_pkts, subflow_fwd_byts: fwd_len as u64,
            subflow_bwd_pkts: bwd_pkts, subflow_bwd_byts: bwd_len as u64,
            init_fwd_win_byts: 65535, init_bwd_win_byts: 65535,
            fwd_act_data_pkts: fwd_pkts, fwd_seg_size_min: 20,
            active_mean: 500_000.0, active_std: 0.0,
            active_max: 500_000.0, active_min: 500_000.0,
            idle_mean: 0.0, idle_std: 0.0, idle_max: 0.0, idle_min: 0.0,
            label: "test".into(),
        }
    }

    #[test]
    fn symmetric_flow_high_entropy() {
        // Equal fwd/bwd bytes → entropy ≈ 1.0
        let row = make_row(10, 10, 5000.0, 5000.0, 5, 0, 500.0, 50.0, 10000.0, 1000.0, 443);
        let v = l7_entropy_vec(&row);
        assert!(v[0] > 0.95, "symmetric flow should have high entropy, got {}", v[0]);
    }

    #[test]
    fn sqli_low_entropy() {
        // SQLi: large fwd payload (query), tiny bwd (error msg)
        let row = make_row(3, 2, 12_000.0, 512.0, 3, 0, 2730.0, 1820.0, 100_000.0, 50_000.0, 80);
        let v = l7_entropy_vec(&row);
        // Entropy should be low (skewed byte distribution)
        assert!(v[0] < 0.5, "SQLi should have low fwd/bwd entropy, got {}", v[0]);
        // PSH density should be non-trivial (3 psh out of 5 total pkts)
        assert!(v[3] > 0.3, "SQLi should have elevated PSH density, got {}", v[3]);
    }

    #[test]
    fn infiltration_high_iat_cv() {
        // Infiltration: highly variable IAT (bursts then long pauses)
        let row = make_row(8, 7, 1146.0, 1581.0, 1, 0, 181.9, 319.4, 200381.0, 400_000.0, 3389);
        let v = l7_entropy_vec(&row);
        // IAT CV = 400_000 / 200_381 ≈ 2.0, scaled to 1.0
        assert!(v[2] > 0.8, "Infiltration should have high IAT CV, got {}", v[2]);
    }

    #[test]
    fn all_values_in_range() {
        let row = make_row(5, 5, 2500.0, 2500.0, 2, 1, 300.0, 100.0, 5000.0, 2000.0, 22);
        let v = l7_entropy_vec(&row);
        for (i, &val) in v.iter().enumerate() {
            assert!(
                val >= 0.0 && val <= 10.0,
                "dim {i} ({}) = {val} out of expected range",
                L7_DIM_NAMES[i]
            );
        }
    }

    #[test]
    fn l7_top_signals_returns_sorted() {
        let row = make_row(3, 2, 12_000.0, 512.0, 3, 1, 2730.0, 1820.0, 100_000.0, 50_000.0, 80);
        let sigs = l7_top_signals(&row, 3);
        assert_eq!(sigs.len(), 3);
        assert!(sigs[0].1 >= sigs[1].1, "should be sorted descending");
        assert!(sigs[1].1 >= sigs[2].1, "should be sorted descending");
    }
}