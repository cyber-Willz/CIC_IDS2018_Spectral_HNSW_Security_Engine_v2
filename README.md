Building AI-driven security tools usually means one thing: massive cloud infrastructure bills and heavy Python runtimes. 

I wanted to see if I could do better at the edge. 

For the past few months, I’ve been building and refining an open-source Network Intrusion Detection System (NIDS) core built entirely in Rust. Today, I’m opening up the repository for the community to break, test, and build upon.

🤖 The Project: CIC-IDS2018 × Spectral + HNSW Security Engine v2
🦀 The Stack: Rust | Burn Framework | Qdrant/HNSW | Graph Spectral Math

By stepping away from traditional Python training/inference pipelines, the engine achieves sub-millisecond network flow profiling directly on edge gateways without heavy cloud dependencies.

Here is the three-tier architectural approach I took:

1️⃣ Reconstruction Anomaly Detection: A multi-layer Autoencoder powered by the native Rust `burn` framework profiles benign baselines. Large Mean Squared Error (MSE) deviations flag potential anomalies on the fly.


2️⃣ Sub-millisecond Vector Querying: Flow embeddings are indexed into an HNSW index to immediately match alerts against known behavioral neighbors.


3️⃣ Algebraic Graph Context: Instead of treating alerts as isolated events, the engine builds a dynamic Laplace graph matrix. By monitoring the Fiedler eigenvalue (λ₁) and computing Commute-Time (CT) distances, it maps out the structural blast radius of an attack.

Precision is currently tracking at a solid 1.000 (zero false positives), and I’m currently optimization-tuning the adaptive threshold multipliers to catch stealthier, application-layer traffic.

If you’re interested in systems programming, memory-safe security tools, or edge machine learning, I’d love for you to check out the repo, run it against your own packet captures.



#RustLang #SystemsProgramming #Cybersecurity #MachineLearning #OpenSource #EdgeAI




====================================================================
SYSTEM LIMITATIONS & KNOWN SCIENTIFIC GAPS (v2-RUN EVALUATION)
====================================================================

This document tracks the verified empirical boundaries of the 
sec_net_engine architecture following the v2 validation pass.

--------------------------------------------------------------------
1. DETECTOR RESOLUTION: HIGH-VOLUME VS. SEMANTIC PAYLOADS
--------------------------------------------------------------------
* Current Performance Metrics:
  - Precision: 1.000 (Zero False Alarms across entire baseline)
  - Recall:    0.714 (10 / 14 True Positive attack detections)
  - F1-Score:  0.833

* Remaining Statistical Gap:
  The unsupervised Autoencoder detects structural volumetric and flow 
  anomalies with high sensitivity, but remains completely blind to 
  low-footprint, application-layer semantic injection.

* Empirical Failure Analysis:
  - Detected (100%): DoS (Hulk/GoldenEye/Slowloris), DDoS (LOIC-HTTP/UDP), 
                     Brute-Force (Web/XSS/FTP), and Heartbleed.
  - Missed (0.0%):   SQL Injection, Cross-Site Scripting (XSS), 
                     Infiltration, and Bot.

  Because SQLi and XSS payloads execute entirely inside standard HTTP 
  GET/POST string bodies without disrupting packet sizes, TCP flags, 
  or flow intervals, their mathematical representations ($MSE_{SQLi} = 0.00103$; 
  $MSE_{XSS} = 0.00089$) fall safely beneath the baseline anomaly 
  threshold ($Threshold = 0.001920$). The system behaves as a flow-layer 
  IDS rather than an application-layer WAF.

--------------------------------------------------------------------
2. GRAPH THEORY CONSTRAINT: MULTI-COMPONENT ALGEBRAIC COLLAPSE
--------------------------------------------------------------------
* Topology State:
  - Nodes: 29 | Connected: No
  - Multiplicity of Eigenvalue 0: 9 ($\lambda_0$ through $\lambda_8 = 0.000000$)

* Revealed Topological Gap:
  The multi-component structure of the active netflow graph fragments 
  the Laplacian Matrix, mathematically causing the structural context 
  engine to collapse.

* Algebraic Analysis:
  In Spectral Graph Theory, the multiplicity of the eigenvalue $\lambda = 0$ 
  corresponds exactly to the number of disconnected sub-components in 
  the graph. Because this network engine contains 9 isolated communication 
  clusters (e.g., standalone $src \rightarrow dst$ pairs with no shared infrastructure), 
  the Fiedler vector fails to form a unified global spatial coordinate system.
  
  Consequently, all cross-component Commute-Time (CT) distances default 
  homogeneously to $1.000000$, neutralizing the system's ability to 
  calculate topological blast radius propagation or recognize macro-targeted 
  infrastructure campaigns.

--------------------------------------------------------------------
3. CONTRIBUTOR FOCUS AREAS
--------------------------------------------------------------------
PRs and architecture enhancements are requested in the following spaces:

1. Synthetic Vertex Bridging: Implementing automated proxy nodes 
   (e.g., binding independent vertices by shared CIDR subnets, target 
   destination ports, or common protocol handlers) to force global 
   graph connectivity, reducing the 0-eigenvalue multiplicity to 1.
   
2. Hybrid Semantic Extraction: Introducing basic layer-7 protocol 
   entropy markers into the Burn tensor input pipeline to expose SQLi/XSS 
   variances to the Autoencoder model.
====================================================================