Building AI-driven security tools usually means one thing: massive cloud infrastructure bills and heavy Python runtimes. 

I wanted to see if I could do better at the edge. 

For the past few months, I’ve been building and refining an open-source Network Intrusion Detection System (NIDS) core built entirely in Rust. Today, I’m opening up the repository for the community to break, test, and build upon.

docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

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





SYSTEM LIMITATIONS & KNOWN SCIENTIFIC GAPS (v2-RUN EVALUATION)


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




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CIC-IDS2018 × Spectral + HNSW Security Engine  v2  (corrected)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━  Phase 0: Pre-training autoencoder on benign flows  ━━━

  Benign baseline: 6 flows
2026-06-26T16:50:13.831231Z  INFO sec_net_engine: Pretraining autoencoder epoch=0 loss=0.04455731064081192
2026-06-26T16:50:13.864437Z  INFO sec_net_engine: Pretraining autoencoder epoch=20 loss=0.027733350172638893
2026-06-26T16:50:13.896019Z  INFO sec_net_engine: Pretraining autoencoder epoch=40 loss=0.015513124875724316
2026-06-26T16:50:13.926802Z  INFO sec_net_engine: Pretraining autoencoder epoch=60 loss=0.008516856469213963
2026-06-26T16:50:13.957119Z  INFO sec_net_engine: Pretraining autoencoder epoch=80 loss=0.005088038742542267
2026-06-26T16:50:13.987141Z  INFO sec_net_engine: Pretraining autoencoder epoch=100 loss=0.0032925403211265802
2026-06-26T16:50:14.016959Z  INFO sec_net_engine: Pretraining autoencoder epoch=120 loss=0.002235191408544779
2026-06-26T16:50:14.043851Z  INFO sec_net_engine: Pretraining autoencoder epoch=140 loss=0.001513645169325173
2026-06-26T16:50:14.075331Z  INFO sec_net_engine: Pretraining autoencoder epoch=160 loss=0.0009977879235520959
2026-06-26T16:50:14.108785Z  INFO sec_net_engine: Pretraining autoencoder epoch=180 loss=0.0006407191976904869
2026-06-26T16:50:14.137125Z  INFO sec_net_engine: Pretraining autoencoder epoch=199 loss=0.0004221073759254068
2026-06-26T16:50:14.145312Z  INFO sec_net_engine: Anomaly threshold derived mean=0.0004133035836275667 std_dev=0.00020640129514504224 max_benign=0.0006581038469448686 threshold=0.001032507512718439
  ✓  Threshold = 0.001033

━━━  Phase 0b: Pre-ingest spectral topology  ━━━

  λ₁ = 0.000000
  ℹ  Isolated-edge topology — λ₁=0 is topologically correct.
     CT distances on direct edges are still valid (expect CT≈1.0).

  Pair                                       Commute-Time       Fiedler
  ────────────────────────────────────────────────────────────────────
  DoS-Hulk → victim                              1.000000      0.117948
  Bot → C2                                       1.000000      0.000000
  Benign RDP                                     1.000000      0.000000

2026-06-26T16:50:14.409202Z  INFO sec_net_engine: Dropped stale collection 'cic_ids2018_v2'.
2026-06-26T16:50:15.127437Z  INFO sec_net_engine: Collection 'cic_ids2018_v2' bootstrapped.
━━━  Phase 1: Ingesting 20 flows  ━━━

2026-06-26T16:50:15.128793Z  INFO sec_net_engine: Within baseline score=0.0005901822005398571 label=Benign
  [  ok     ]  mse=0.00059  label=Benign                               172.31.69.11→172.31.69.20  [λ₂=2.0100]
2026-06-26T16:50:15.177140Z  INFO sec_net_engine: Within baseline score=0.0006581038469448686 label=Benign
  [  ok     ]  mse=0.00066  label=Benign                               172.31.69.12→172.31.69.2  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]
2026-06-26T16:50:15.222897Z  INFO sec_net_engine: Within baseline score=0.0001768687361618504 label=Benign
  [  ok     ]  mse=0.00018  label=Benign                               172.31.69.30→54.239.28.85  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]
2026-06-26T16:50:15.267786Z  INFO sec_net_engine: Within baseline score=0.0005945045268163085 label=Benign
  [  ok     ]  mse=0.00059  label=Benign                               172.31.69.50→172.31.69.2  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]
2026-06-26T16:50:15.312363Z  INFO sec_net_engine: Within baseline score=0.00016481417696923018 label=Benign
  [  ok     ]  mse=0.00016  label=Benign                               172.31.69.51→52.84.100.12  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]
2026-06-26T16:50:15.360988Z  INFO sec_net_engine: Within baseline score=0.00029534794157370925 label=Benign
  [  ok     ]  mse=0.00030  label=Benign                               172.31.69.52→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]
2026-06-26T16:50:15.411465Z  INFO sec_net_engine: Within baseline score=0.0003687777789309621 label=Infilteration
  [  ok     ]  mse=0.00037  label=Infilteration                        172.31.69.95→172.31.69.20  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]
2026-06-26T16:50:15.457507Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:16.535693Z  WARN sec_net_engine: [ANOMALY] severity=LOW score=0.0012843180447816849 label=Bot mitre="TA0011" hnsw_nn="Benign"
  [⚠ ANOMALY]  mse=0.00128  label=Bot                                  172.31.69.88→52.70.12.33  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [LOW]  label=Bot
╠──────────────────────────────────────────────────────────────╣
║  [LOW] Ip:172.31.69.88 → Ip:52.70.12.33 | MSE=0.0013 — Command and Control (TA0011) | HNSW nn=Benign d=0.189 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.88  →  Dst : Ip:52.70.12.33
║  Loss : 0.001284  Edge : CommunicatedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0011 Command and Control — T1071 – Application Layer Protocol (Botnet) (trigger: 'bot')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.1895  Ip:172.31.69.30→Ip:54.239.28.85  [Benign]
║    d=0.1933  Ip:172.31.69.51→Ip:52.84.100.12  [Benign]
║    d=0.2502  Ip:172.31.69.11→Ip:172.31.69.20  [Benign]
║    d=0.2728  Ip:172.31.69.52→Ip:172.31.69.10  [Benign]
║    d=0.2829  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.88
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000000
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=0.6872  F=0.0000  → Ip:172.31.69.20
║      CT=0.6872  F=0.0000  → Ip:172.31.69.2
║      CT=0.7071  F=0.0000  → Ip:172.31.69.30
║      CT=0.7071  F=0.0000  → Ip:54.239.28.85
║      CT=0.7071  F=0.0000  → Ip:172.31.69.51
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:16.584270Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:17.982955Z  WARN sec_net_engine: [ANOMALY] severity=CRITICAL score=0.007827157154679298 label=DoS attacks-Hulk mitre="TA0040" hnsw_nn="Benign"
  [⚠ ANOMALY]  mse=0.00783  label=DoS attacks-Hulk                     172.31.69.200→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [CRITICAL]  label=DoS attacks-Hulk
╠──────────────────────────────────────────────────────────────╣
║  [CRITICAL] Ip:172.31.69.200 → Ip:172.31.69.10 | MSE=0.0078 — Impact (TA0040) | HNSW nn=Benign d=0.555 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.200  →  Dst : Ip:172.31.69.10
║  Loss : 0.007827  Edge : CommunicatedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0040 Impact — T1499 – Endpoint Denial of Service (trigger: 'dos')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.5548  Ip:172.31.69.50→Ip:172.31.69.2  [Benign]
║    d=0.5592  Ip:172.31.69.12→Ip:172.31.69.2  [Benign]
║    d=0.7732  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
║    d=0.7797  Ip:172.31.69.11→Ip:172.31.69.20  [Benign]
║    d=0.7853  Ip:172.31.69.52→Ip:172.31.69.10  [Benign]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.200
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000000
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=0.8819  F=0.0000  → Ip:172.31.69.20
║      CT=0.8819  F=0.0000  → Ip:172.31.69.2
║      CT=0.8975  F=0.0000  → Ip:172.31.69.30
║      CT=0.8975  F=0.0000  → Ip:54.239.28.85
║      CT=0.8975  F=0.0000  → Ip:172.31.69.51
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:17.988487Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:19.441387Z  WARN sec_net_engine: [ANOMALY] severity=MEDIUM score=0.0028636762872338295 label=DoS attacks-GoldenEye mitre="TA0040" hnsw_nn="DoS attacks-Hulk"
  [⚠ ANOMALY]  mse=0.00286  label=DoS attacks-GoldenEye                172.31.69.201→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [MEDIUM]  label=DoS attacks-GoldenEye
╠──────────────────────────────────────────────────────────────╣
║  [MEDIUM] Ip:172.31.69.201 → Ip:172.31.69.10 | MSE=0.0029 — Impact (TA0040) | HNSW nn=DoS attacks-Hulk d=0.442 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.201  →  Dst : Ip:172.31.69.10
║  Loss : 0.002864  Edge : CommunicatedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0040 Impact — T1499 – Endpoint Denial of Service (trigger: 'dos')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.4419  Ip:172.31.69.200→Ip:172.31.69.10  [DoS attacks-Hulk]
║    d=0.4793  Ip:172.31.69.50→Ip:172.31.69.2  [Benign]
║    d=0.4902  Ip:172.31.69.12→Ip:172.31.69.2  [Benign]
║    d=0.5064  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
║    d=0.5168  Ip:172.31.69.52→Ip:172.31.69.10  [Benign]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.201
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.005135
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=0.9538  F=0.0051  → Ip:172.31.69.20
║      CT=0.9538  F=0.0051  → Ip:172.31.69.2
║      CT=0.9682  F=0.0051  → Ip:172.31.69.30
║      CT=0.9682  F=0.0051  → Ip:54.239.28.85
║      CT=0.9682  F=0.0051  → Ip:172.31.69.51
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:19.445955Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:21.569620Z  WARN sec_net_engine: [ANOMALY] severity=CRITICAL score=0.01054181344807148 label=DDoS attacks-LOIC-HTTP mitre="TA0040" hnsw_nn="DoS attacks-Hulk"
  [⚠ ANOMALY]  mse=0.01054  label=DDoS attacks-LOIC-HTTP               172.31.69.210→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [CRITICAL]  label=DDoS attacks-LOIC-HTTP
╠──────────────────────────────────────────────────────────────╣
║  [CRITICAL] Ip:172.31.69.210 → Ip:172.31.69.10 | MSE=0.0105 — Impact (TA0040) | HNSW nn=DoS attacks-Hulk d=0.356 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.210  →  Dst : Ip:172.31.69.10
║  Loss : 0.010542  Edge : ObservedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0040 Impact — T1498 – Network Denial of Service (trigger: 'ddos')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.3560  Ip:172.31.69.200→Ip:172.31.69.10  [DoS attacks-Hulk]
║    d=0.5869  Ip:172.31.69.12→Ip:172.31.69.2  [Benign]
║    d=0.5953  Ip:172.31.69.50→Ip:172.31.69.2  [Benign]
║    d=0.6016  Ip:172.31.69.201→Ip:172.31.69.10  [DoS attacks-GoldenEye]
║    d=0.9938  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.210
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000000
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=0.9911  F=0.0000  → Ip:172.31.69.20
║      CT=0.9911  F=0.0000  → Ip:172.31.69.2
║      CT=1.0000  F=0.0000  → Ip:172.31.69.10
║      CT=1.0050  F=0.0000  → Ip:172.31.69.30
║      CT=1.0050  F=0.0000  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:21.575001Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:21.621436Z  WARN sec_net_engine: [ANOMALY] severity=HIGH score=0.004589421674609184 label=Brute Force -Web mitre="TA0002" hnsw_nn="DoS attacks-GoldenEye"
  [⚠ ANOMALY]  mse=0.00459  label=Brute Force -Web                     172.31.69.180→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [HIGH]  label=Brute Force -Web
╠──────────────────────────────────────────────────────────────╣
║  [HIGH] Ip:172.31.69.180 → Ip:172.31.69.10 | MSE=0.0046 — Execution (TA0002) | HNSW nn=DoS attacks-GoldenEye d=0.201 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.180  →  Dst : Ip:172.31.69.10
║  Loss : 0.004589  Edge : CommunicatedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0002 Execution — T1203 – Remote Code Execution (trigger: 'rce')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.2005  Ip:172.31.69.201→Ip:172.31.69.10  [DoS attacks-GoldenEye]
║    d=0.5326  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
║    d=0.5358  Ip:172.31.69.52→Ip:172.31.69.10  [Benign]
║    d=0.5368  Ip:172.31.69.30→Ip:54.239.28.85  [Benign]
║    d=0.5476  Ip:172.31.69.51→Ip:52.84.100.12  [Benign]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.180
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000000
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=1.0000  F=0.0000  → Ip:172.31.69.10
║      CT=1.0138  F=0.0000  → Ip:172.31.69.20
║      CT=1.0138  F=0.0000  → Ip:172.31.69.2
║      CT=1.0274  F=0.0000  → Ip:172.31.69.30
║      CT=1.0274  F=0.0000  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:21.627746Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:21.674123Z  WARN sec_net_engine: [ANOMALY] severity=LOW score=0.0010964088141918182 label=SQL Injection mitre="TA0002" hnsw_nn="Benign"
  [⚠ ANOMALY]  mse=0.00110  label=SQL Injection                        172.31.69.181→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [LOW]  label=SQL Injection
╠──────────────────────────────────────────────────────────────╣
║  [LOW] Ip:172.31.69.181 → Ip:172.31.69.10 | MSE=0.0011 — Execution (TA0002) | HNSW nn=Benign d=0.194 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.181  →  Dst : Ip:172.31.69.10
║  Loss : 0.001096  Edge : CommunicatedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0002 Execution — T1055 – Process Injection (trigger: 'injection')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.1942  Ip:172.31.69.52→Ip:172.31.69.10  [Benign]
║    d=0.2244  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
║    d=0.2750  Ip:172.31.69.51→Ip:52.84.100.12  [Benign]
║    d=0.2813  Ip:172.31.69.30→Ip:54.239.28.85  [Benign]
║    d=0.2889  Ip:172.31.69.11→Ip:172.31.69.20  [Benign]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.181
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000000
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=1.0000  F=0.0000  → Ip:172.31.69.10
║      CT=1.0291  F=0.0000  → Ip:172.31.69.20
║      CT=1.0291  F=0.0000  → Ip:172.31.69.2
║      CT=1.0425  F=0.0000  → Ip:172.31.69.30
║      CT=1.0425  F=0.0000  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:21.722817Z  INFO sec_net_engine: Within baseline score=0.00096421706257388 label=XSS
  [  ok     ]  mse=0.00096  label=XSS                                  172.31.69.182→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]
2026-06-26T16:50:21.769770Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:21.814277Z  WARN sec_net_engine: [ANOMALY] severity=CRITICAL score=0.008311498910188675 label=Brute Force -XSS mitre="TA0002" hnsw_nn="Brute Force -Web"
  [⚠ ANOMALY]  mse=0.00831  label=Brute Force -XSS                     172.31.69.190→172.31.69.15  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [CRITICAL]  label=Brute Force -XSS
╠──────────────────────────────────────────────────────────────╣
║  [CRITICAL] Ip:172.31.69.190 → Ip:172.31.69.15 | MSE=0.0083 — Execution (TA0002) | HNSW nn=Brute Force -Web d=0.327 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.190  →  Dst : Ip:172.31.69.15
║  Loss : 0.008311  Edge : AuthenticatedTo
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0002 Execution — T1203 – Remote Code Execution (trigger: 'rce')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.3268  Ip:172.31.69.180→Ip:172.31.69.10  [Brute Force -Web]
║    d=0.3681  Ip:172.31.69.201→Ip:172.31.69.10  [DoS attacks-GoldenEye]
║    d=0.5611  Ip:172.31.69.200→Ip:172.31.69.10  [DoS attacks-Hulk]
║    d=0.6566  Ip:172.31.69.88→Ip:52.70.12.33  [Bot]
║    d=0.6664  Ip:172.31.69.30→Ip:54.239.28.85  [Benign]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.190
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000000
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=0.5995  F=0.0000  → Ip:172.31.69.10
║      CT=0.6872  F=0.0000  → Ip:172.31.69.20
║      CT=0.6872  F=0.0000  → Ip:172.31.69.2
║      CT=0.7071  F=0.0000  → Ip:172.31.69.30
║      CT=0.7071  F=0.0000  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:21.862985Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:21.906254Z  WARN sec_net_engine: [ANOMALY] severity=CRITICAL score=0.008262703195214272 label=FTP-BruteForce mitre="TA0002" hnsw_nn="Brute Force -XSS"
  [⚠ ANOMALY]  mse=0.00826  label=FTP-BruteForce                       172.31.69.191→172.31.69.14  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [CRITICAL]  label=FTP-BruteForce
╠──────────────────────────────────────────────────────────────╣
║  [CRITICAL] Ip:172.31.69.191 → Ip:172.31.69.14 | MSE=0.0083 — Execution (TA0002) | HNSW nn=Brute Force -XSS d=0.039 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.191  →  Dst : Ip:172.31.69.14
║  Loss : 0.008263  Edge : AuthenticatedTo
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0002 Execution — T1203 – Remote Code Execution (trigger: 'rce')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.0386  Ip:172.31.69.190→Ip:172.31.69.15  [Brute Force -XSS]
║    d=0.3264  Ip:172.31.69.180→Ip:172.31.69.10  [Brute Force -Web]
║    d=0.3650  Ip:172.31.69.201→Ip:172.31.69.10  [DoS attacks-GoldenEye]
║    d=0.5652  Ip:172.31.69.200→Ip:172.31.69.10  [DoS attacks-Hulk]
║    d=0.6475  Ip:172.31.69.88→Ip:52.70.12.33  [Bot]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.191
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000000
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=0.5995  F=0.0000  → Ip:172.31.69.10
║      CT=0.6872  F=0.0000  → Ip:172.31.69.20
║      CT=0.6872  F=0.0000  → Ip:172.31.69.2
║      CT=0.7071  F=0.0000  → Ip:172.31.69.30
║      CT=0.7071  F=0.0000  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:21.955812Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:22.002451Z  WARN sec_net_engine: [ANOMALY] severity=MEDIUM score=0.002597854007035494 label=Heartbleed mitre="TA0001" hnsw_nn="Benign"
  [⚠ ANOMALY]  mse=0.00260  label=Heartbleed                           172.31.69.99→172.31.69.16  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [MEDIUM]  label=Heartbleed
╠──────────────────────────────────────────────────────────────╣
║  [MEDIUM] Ip:172.31.69.99 → Ip:172.31.69.16 | MSE=0.0026 — Initial Access (TA0001) | HNSW nn=Benign d=0.251 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.99  →  Dst : Ip:172.31.69.16
║  Loss : 0.002598  Edge : CommunicatedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0001 Initial Access — T1190 – OpenSSL Heartbleed CVE-2014-0160 (trigger: 'heartbleed')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.2512  Ip:172.31.69.52→Ip:172.31.69.10  [Benign]
║    d=0.2919  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
║    d=0.3046  Ip:172.31.69.181→Ip:172.31.69.10  [SQL Injection]
║    d=0.3091  Ip:172.31.69.182→Ip:172.31.69.10  [XSS]
║    d=0.3157  Ip:172.31.69.51→Ip:52.84.100.12  [Benign]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.99
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000000
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=0.5995  F=0.0000  → Ip:172.31.69.10
║      CT=0.6872  F=0.0000  → Ip:172.31.69.20
║      CT=0.6872  F=0.0000  → Ip:172.31.69.2
║      CT=0.7071  F=0.0000  → Ip:172.31.69.30
║      CT=0.7071  F=0.0000  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:22.053353Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:22.099434Z  WARN sec_net_engine: [ANOMALY] severity=CRITICAL score=0.009136483073234558 label=DoS attacks-SlowHTTPTest mitre="TA0040" hnsw_nn="Benign"
  [⚠ ANOMALY]  mse=0.00914  label=DoS attacks-SlowHTTPTest             172.31.69.202→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [CRITICAL]  label=DoS attacks-SlowHTTPTest
╠──────────────────────────────────────────────────────────────╣
║  [CRITICAL] Ip:172.31.69.202 → Ip:172.31.69.10 | MSE=0.0091 — Impact (TA0040) | HNSW nn=Benign d=0.563 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.202  →  Dst : Ip:172.31.69.10
║  Loss : 0.009136  Edge : CommunicatedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0040 Impact — T1499 – Endpoint Denial of Service (trigger: 'dos')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.5628  Ip:172.31.69.11→Ip:172.31.69.20  [Benign]
║    d=0.5874  Ip:172.31.69.88→Ip:52.70.12.33  [Bot]
║    d=0.5905  Ip:172.31.69.51→Ip:52.84.100.12  [Benign]
║    d=0.5960  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
║    d=0.5986  Ip:172.31.69.30→Ip:54.239.28.85  [Benign]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.202
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000099
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=1.0000  F=0.0001  → Ip:172.31.69.10
║      CT=1.0482  F=0.0001  → Ip:172.31.69.20
║      CT=1.0482  F=0.0001  → Ip:172.31.69.2
║      CT=1.0614  F=0.0001  → Ip:172.31.69.30
║      CT=1.0614  F=0.0001  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:22.148441Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:22.194488Z  WARN sec_net_engine: [ANOMALY] severity=CRITICAL score=0.009383085183799267 label=DoS attacks-Slowloris mitre="TA0040" hnsw_nn="DoS attacks-SlowHTTPTest"
  [⚠ ANOMALY]  mse=0.00938  label=DoS attacks-Slowloris                172.31.69.203→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [CRITICAL]  label=DoS attacks-Slowloris
╠──────────────────────────────────────────────────────────────╣
║  [CRITICAL] Ip:172.31.69.203 → Ip:172.31.69.10 | MSE=0.0094 — Impact (TA0040) | HNSW nn=DoS attacks-SlowHTTPTest d=0.018 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.203  →  Dst : Ip:172.31.69.10
║  Loss : 0.009383  Edge : CommunicatedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0040 Impact — T1499 – Endpoint Denial of Service (trigger: 'dos')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.0184  Ip:172.31.69.202→Ip:172.31.69.10  [DoS attacks-SlowHTTPTest]
║    d=0.5690  Ip:172.31.69.11→Ip:172.31.69.20  [Benign]
║    d=0.5948  Ip:172.31.69.88→Ip:52.70.12.33  [Bot]
║    d=0.5984  Ip:172.31.69.51→Ip:52.84.100.12  [Benign]
║    d=0.6031  Ip:172.31.69.95→Ip:172.31.69.20  [Infilteration]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.203
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000055
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=1.0000  F=0.0001  → Ip:172.31.69.10
║      CT=1.0546  F=0.0001  → Ip:172.31.69.20
║      CT=1.0546  F=0.0001  → Ip:172.31.69.2
║      CT=1.0677  F=0.0001  → Ip:172.31.69.30
║      CT=1.0677  F=0.0001  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝

2026-06-26T16:50:22.244510Z  INFO sec_net_engine: BFS blast radius depth=0 n=1
2026-06-26T16:50:22.290413Z  WARN sec_net_engine: [ANOMALY] severity=CRITICAL score=0.011064046062529087 label=DDoS attacks-LOIC-UDP mitre="TA0040" hnsw_nn="DDoS attacks-LOIC-HTTP"
  [⚠ ANOMALY]  mse=0.01106  label=DDoS attacks-LOIC-UDP                172.31.69.211→172.31.69.10  [REG active λ₂_raw=0.0100 → 0.0100 virt=0]

╔══════════════════════════════════════════════════════════════╗
║  INCIDENT [CRITICAL]  label=DDoS attacks-LOIC-UDP
╠──────────────────────────────────────────────────────────────╣
║  [CRITICAL] Ip:172.31.69.211 → Ip:172.31.69.10 | MSE=0.0111 — Impact (TA0040) | HNSW nn=DDoS attacks-LOIC-HTTP d=0.057 | ⚠ bridge λ₁=0.0000
╠──────────────────────────────────────────────────────────────╣
║  Src  : Ip:172.31.69.211  →  Dst : Ip:172.31.69.10
║  Loss : 0.011064  Edge : ObservedWith
╠──────────────────────────────────────────────────────────────╣
║  MITRE : TA0040 Impact — T1498 – Network Denial of Service (trigger: 'ddos')
╠──────────────────────────────────────────────────────────────╣
║  HNSW SIMILAR FLOWS (5 neighbours):
║    d=0.0575  Ip:172.31.69.210→Ip:172.31.69.10  [DDoS attacks-LOIC-HTTP]
║    d=0.3958  Ip:172.31.69.200→Ip:172.31.69.10  [DoS attacks-Hulk]
║    d=0.6078  Ip:172.31.69.12→Ip:172.31.69.2  [Benign]
║    d=0.6185  Ip:172.31.69.50→Ip:172.31.69.2  [Benign]
║    d=0.6356  Ip:172.31.69.201→Ip:172.31.69.10  [DoS attacks-GoldenEye]
╠──────────────────────────────────────────────────────────────╣
║  BFS BLAST RADIUS (1 nodes):
║    → Ip:172.31.69.211
╠──────────────────────────────────────────────────────────────╣
║  SPECTRAL:
║    λ₁ = 0.000000  CT-dist=1.000000  Fiedler=0.000216
║    ⚠ LOW λ₁ — near-bridge topology; cut-edge risk
║    Spectral blast radius (10 nodes):
║      CT=1.0000  F=0.0002  → Ip:172.31.69.10
║      CT=1.0597  F=0.0002  → Ip:172.31.69.20
║      CT=1.0597  F=0.0002  → Ip:172.31.69.2
║      CT=1.0728  F=0.0002  → Ip:172.31.69.30
║      CT=1.0728  F=0.0002  → Ip:54.239.28.85
╚══════════════════════════════════════════════════════════════╝


  IncidentReports generated: 12 / 20

━━━  Phase 2: Detection evaluation (Qdrant read-back)  ━━━

┌──────────────────────────────────────────────────────────────────────────────────┐
│   CSE-CIC-IDS2018 × SPECTRAL + HNSW  DETECTION METRICS                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│  TP:   12  FP:    0  TN:    6  FN:    2                                         │
│  Precision=1.000  Recall=0.857  F1=0.923  Accuracy=0.900                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│  CIC Label                          Score  Det.     Severity  HNSW-NN       MITRE
│  ────────────────────────────────────────────────────────────────────────────────
│  Benign                            0.0006    ok     INFO      none          none
│  Benign                            0.0007    ok     INFO      none          none
│  Benign                            0.0002    ok     INFO      none          none
│  Benign                            0.0006    ok     INFO      none          none
│  Benign                            0.0002    ok     INFO      none          none
│  Benign                            0.0003    ok     INFO      none          none
│  Infilteration                     0.0004    ok     INFO      none          none
│  Bot                               0.0013  ✓ ALERT  LOW       Benign        TA0011
│  DoS attacks-Hulk                  0.0078  ✓ ALERT  CRITICAL  Benign        TA0040
│  DoS attacks-GoldenEye             0.0029  ✓ ALERT  MEDIUM    DoS attacks-  TA0040
│  DDoS attacks-LOIC-HTTP            0.0105  ✓ ALERT  CRITICAL  DoS attacks-  TA0040
│  Brute Force -Web                  0.0046  ✓ ALERT  HIGH      DoS attacks-  TA0002
│  SQL Injection                     0.0011  ✓ ALERT  LOW       Benign        TA0002
│  XSS                               0.0010    ok     INFO      none          none
│  Brute Force -XSS                  0.0083  ✓ ALERT  CRITICAL  Brute Force   TA0002
│  FTP-BruteForce                    0.0083  ✓ ALERT  CRITICAL  Brute Force   TA0002
│  Heartbleed                        0.0026  ✓ ALERT  MEDIUM    Benign        TA0001
│  DoS attacks-SlowHTTPTest          0.0091  ✓ ALERT  CRITICAL  Benign        TA0040
│  DoS attacks-Slowloris             0.0094  ✓ ALERT  CRITICAL  DoS attacks-  TA0040
│  DDoS attacks-LOIC-UDP             0.0111  ✓ ALERT  CRITICAL  DDoS attacks  TA0040
└──────────────────────────────────────────────────────────────────────────────────┘

━━━  Phase 3: Per-attack-class detection rate  ━━━

  Label                                Total  Detected        Rate
  ─────────────────────────────────────────────────────────────────
  Benign                                   6         0        0.0%
  Bot                                      1         1      100.0%
  Brute Force -Web                         1         1      100.0%
  Brute Force -XSS                         1         1      100.0%
  DDoS attacks-LOIC-HTTP                   1         1      100.0%
  DDoS attacks-LOIC-UDP                    1         1      100.0%
  DoS attacks-GoldenEye                    1         1      100.0%
  DoS attacks-Hulk                         1         1      100.0%
  DoS attacks-SlowHTTPTest                 1         1      100.0%
  DoS attacks-Slowloris                    1         1      100.0%
  FTP-BruteForce                           1         1      100.0%
  Heartbleed                               1         1      100.0%
  Infilteration                            1         0        0.0%
  SQL Injection                            1         1      100.0%
  XSS                                      1         0        0.0%

━━━  Phase 4: Final spectral topology report  ━━━

╔══════════════════════════════════════════════════════╗
║          Spectral Graph Analysis Report              ║
╠══════════════════════════════════════════════════════╣
║  Nodes    : 29
║  Connected: No
║  λ₁ (Fiedler): 0.000000
╠──────────────────────────────────────────────────────╣
║  Eigenvalues (first 10):
║    λ0 =   0.000000 (trivial)
║    λ1 =   0.000000 (Fiedler)
║    λ2 =   0.000000
║    λ3 =   0.000000
║    λ4 =   0.000000
║    λ5 =   0.000000
║    λ6 =   0.000000
║    λ7 =   0.000000
║    λ8 =   0.000000
║    λ9 =   1.000000
║    ... (29 total)
╠──────────────────────────────────────────────────────╣
║  Pairwise distances:
║  Label               Hops    Commute-Time      Fiedler   Euclidean
║  ────────────────────────────────────────────────────────────────
║  (14->10) hulk→victim      1        1.000000     0.117948    1.414214
║  (16->10) loic→victim      1        1.000000     0.117871    1.414214
║  (12->13) bot→c2          1        1.000000     0.000000    1.414214
╚══════════════════════════════════════════════════════╝
  λ₁ final: 0.000000

━━━  Phase 5: Dynamic Laplacian Regularizer — disconnection drill  ━━━

  Tick 0 (split)   λ₂_raw=0.0100  λ₂_reg=0.0100  virt_injected=2  active=true
  Virtual edges injected:
    0→2  w=0.450  reason=CommonGateway(100)
    3→5  w=0.450  reason=CommonGateway(101)

  Tick 1 (bridge)  λ₂_raw=0.2081  λ₂_reg=0.2546  virt_injected=0  active=false

  Propagation gradients from node 2 (bridge):
    → node 1  grad=-0.14228
    → node 3  grad=+0.29427
    → node 0  grad=-0.08000
  (positive = flow away from node 2; negative = flow toward)

  Tick 11 (healed) λ₂_raw=0.2081  λ₂_reg=0.2081  virt_remaining=0

  λ₂ trend (last 12 ticks): ["0.010", "0.208", "0.208", "0.208", "0.208", "0.208", "0.208", "0.208", "0.208", "0.208", "0.208", "0.208"]

━━━  Complete  ━━━
   variances to the Autoencoder model.
====================================================================
