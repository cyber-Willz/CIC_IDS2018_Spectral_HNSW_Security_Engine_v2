//! Pure-Rust spectral graph library used by the security engine.
//!
//! ## Modules
//! - [`graph`]     — undirected adjacency-list graph + BFS
//! - [`embedding`] — Jacobi eigendecomposition → spectral coordinates
//! - [`report`]    — pretty-printable pairwise distance report
//! - [`error`]     — shared error / result types

pub mod embedding;
pub mod error;
pub mod graph;
pub mod report;

pub use embedding::{JacobiConfig, SpectralEmbedding};
pub use error::{GraphError, GraphResult};
pub use graph::Graph;
pub use report::{GraphReport, PairwiseResult};