use std::collections::VecDeque;

use log::debug;
use serde::{Deserialize, Serialize};

use super::error::{GraphError, GraphResult};

/// Strongly-typed, copy-cheap node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Undirected, unweighted graph stored as a dense adjacency list.
///
/// All mutations validate arguments and return [`GraphError`] rather than
/// panicking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    num_nodes: usize,
    adjacency: Vec<Vec<usize>>,
}

impl Graph {
    /// Create an empty graph with `num_nodes` nodes (no edges).
    pub fn new(num_nodes: usize) -> GraphResult<Self> {
        if num_nodes == 0 {
            return Err(GraphError::EmptyGraph);
        }
        Ok(Graph { num_nodes, adjacency: vec![Vec::new(); num_nodes] })
    }

    /// Add an undirected edge `u — v`.  Self-loops are silently dropped;
    /// duplicate edges are ignored.
    pub fn add_edge(&mut self, u: usize, v: usize) -> GraphResult<()> {
        self.validate_node(u)?;
        self.validate_node(v)?;
        if u == v {
            debug!("Skipping self-loop on node {u}");
            return Ok(());
        }
        if !self.adjacency[u].contains(&v) {
            self.adjacency[u].push(v);
            self.adjacency[v].push(u);
        }
        Ok(())
    }

    /// Build a graph from a slice of `(u, v)` edge pairs.
    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize)]) -> GraphResult<Self> {
        let mut g = Graph::new(num_nodes)?;
        for &(u, v) in edges {
            g.add_edge(u, v)?;
        }
        Ok(g)
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn degree(&self, idx: usize) -> GraphResult<usize> {
        self.validate_node(idx)?;
        Ok(self.adjacency[idx].len())
    }

    pub fn neighbours(&self, idx: usize) -> GraphResult<&[usize]> {
        self.validate_node(idx)?;
        Ok(&self.adjacency[idx])
    }

    /// BFS shortest-path hop distance between `start` and `end`.
    pub fn topological_distance(&self, start: usize, end: usize) -> GraphResult<usize> {
        self.validate_node(start)?;
        self.validate_node(end)?;
        if start == end {
            return Ok(0);
        }

        let mut dist = vec![usize::MAX; self.num_nodes];
        dist[start] = 0;
        let mut queue = VecDeque::with_capacity(self.num_nodes);
        queue.push_back(start);

        while let Some(curr) = queue.pop_front() {
            let d = dist[curr];
            if curr == end {
                return Ok(d);
            }
            for &nbr in &self.adjacency[curr] {
                if dist[nbr] == usize::MAX {
                    dist[nbr] = d + 1;
                    queue.push_back(nbr);
                }
            }
        }
        Err(GraphError::NoPath(start, end))
    }

    /// Full BFS distance vector from `start`.  Unreachable nodes map to `None`.
    pub fn bfs_distances(&self, start: usize) -> GraphResult<Vec<Option<usize>>> {
        self.validate_node(start)?;
        let mut dist = vec![None; self.num_nodes];
        dist[start] = Some(0);
        let mut queue = VecDeque::with_capacity(self.num_nodes);
        queue.push_back(start);
        while let Some(curr) = queue.pop_front() {
            let d = dist[curr].unwrap();
            for &nbr in &self.adjacency[curr] {
                if dist[nbr].is_none() {
                    dist[nbr] = Some(d + 1);
                    queue.push_back(nbr);
                }
            }
        }
        Ok(dist)
    }

    /// Unnormalised Laplacian **L = D − A** as a dense row-major matrix.
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.num_nodes;
        let mut l = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            l[i][i] = self.adjacency[i].len() as f64;
            for &j in &self.adjacency[i] {
                l[i][j] = -1.0;
            }
        }
        l
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

impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph ({} nodes):", self.num_nodes)?;
        for (i, nbrs) in self.adjacency.iter().enumerate() {
            write!(f, "  Node {i}: [")?;
            for (k, &nbr) in nbrs.iter().enumerate() {
                if k > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{nbr}")?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}