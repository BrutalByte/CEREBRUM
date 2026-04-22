# CEREBRUM System Architecture

**Version**: v2.21.0 (Phase 94 COMPLETE)

Complete data-flow from ingestion to result, including all options, pathways, and decision nodes.

```mermaid
flowchart TD

%% ═══════════════════════════════════════════════════════════
%% ENTRY POINTS
%% ═══════════════════════════════════════════════════════════

subgraph ENTRY["① Entry Points"]
    direction LR
    E_REST["REST API\n/query  /feedback\n/retrain  /params"]
    E_CLI["CLI\ncerebrum query\ncerebrum serve"]
    E_UI["Gradio Studio\nui/studio.py"]
    E_FED["Federated Node\nPOST /traverse"]
    E_STREAM["Live Stream\n/stream/ingest"]
end

%% ═══════════════════════════════════════════════════════════
%% INGESTION — THALAMUS
%% ═══════════════════════════════════════════════════════════

subgraph THALAMUS["② THALAMUS — Ingestion Pipeline"]
    direction TB

    subgraph ADAPTERS["Graph Source"]
        direction LR
        A_CSV["CSV\ncsvadapter"]
        A_FILE["GraphML / JSON\nfile_adapter"]
        A_NX["NetworkX\nnetworkx_adapter"]
        A_NEO["Neo4j\nneo4j_adapter"]
        A_RDF["RDF / SPARQL\nrdf_adapter"]
        A_FED2["Federated\nfederated_adapter"]
        A_STREAM2["Stream\nstream_adapter"]
    end

    D_PIPE{"IngestionPipeline\nenabled?"}
    PIPE["IngestionPipeline\nnormalize entities\ndedup aliases\nnorm relations\nconfidence + provenance"]

    D_STDP{"STDPDiscretizer\nenabled?"}
    STDP["STDPDiscretizer\ncausal edge inference\nchi-squared filter\nCausalSignificanceFilter"]

    D_SIG{"SignalEncoder\nenabled?"}
    SIG["SignalEncoder\nStatistical / Spectral\nProcrustes SVD → entity space\nnamespace isolation"]

    D_COMPLETE{"graph.complete()\nrules?"}
    COMPLETE["Graph Completion\nInverseRule\nCompositionRule\nsynth edges + provenance"]

    D_ENHANCE{"graph.enhance()\nenhancers?"}
    ENHANCE["Graph Enhancement\nGraphBridgeEngine\nheuristic link synthesis"]

    NX_ADAPTER["NetworkXAdapter\nunified graph object\nEntity + Edge model"]
end

%% ═══════════════════════════════════════════════════════════
%% BUILD PIPELINE
%% ═══════════════════════════════════════════════════════════

subgraph BUILD["③ CerebrumGraph.build() — THALAMUS Pipeline"]
    direction TB

    CACHE_CHK{"embeddings.pkl\ncache hit?"}

    subgraph EMB_BLOCK["Embedding Step"]
        D_EMB{"embedding mode"}
        EMB_RAND["RandomEngine\nseeded hash vectors\n64-dim  fast"]
        EMB_SENT["SentenceEngine\nBGE-Small-v1.5\n384-dim  instruction-tuned"]
        EMB_SAVE["save embeddings.pkl"]
    end

    D_SAGE{"use_graphsage?"}
    SAGE_CACHE{"embeddings_sage.pkl\ncache hit?"}
    SAGE["GraphSAGE Smoother\none-pass mean aggregation\nself + neighbour blend\nsave embeddings_sage.pkl"]

    STRUCT["StructuralEncoder\nPageRank · betweenness · degree\nnode_recency score\nresidual + LayerNorm → embeddings"]

    subgraph COMM_BLOCK["Community Detection"]
        D_COMM{"community_engine"}
        C_DSCF["DSCF\nDual-Signal Community Fusion\nbest-of-N trials"]
        C_TSC["TSC\nTriple-Signal Consensus\nauto PageRank weights"]
        C_LEIDEN["Leiden\nfast for large graphs"]
        C_LPA["LPA\nLabel Propagation"]
        D_COARSEN{"coarsen?"}
        COARSEN["coarsen_communities\nmin_size or target_max"]
        COMM_SAVE["save communities.pkl"]
    end

    CSA_BUILD["CSAEngine init\n10-param attention formula\ncommunity distance matrix\nadjacent pairs\ncommunity graph\nPageRank prior loaded"]

    D_PROB{"probabilistic\nmode?"}
    BT_PROB["BeamTraversal\nprobabilistic=True\nBeta distribution paths\nThompson sampling\nwarm_start_strength"]
    BT_STD["BeamTraversal\ndeterministic\nscore-sorted pruning"]

    D_ENGRAM_CACHE{"Engram\npath provided?"}
    BT_ENGRAM["EngramTraversal\n_prune_candidates override\naffinity-boosted scoring\nrelation-pattern steering"]
end

%% ═══════════════════════════════════════════════════════════
%% QUERY / CORTEX
%% ═══════════════════════════════════════════════════════════

subgraph CORTEX["④ CORTEX — Query Execution"]
    direction TB

    SEED_FIND["find_entities(query, top_k=1)\nfuzzy seed resolution"]

    D_QUERY_EMB{"query_embedding\nprovided?"}
    Q_EMB["encode_query()\nBGE instruction prefix\nquery vector"]
    Q_SNAPSHOT["CSA.set_query_snapshot()\ncommunity map frozen\nprevents mid-flight swap"]

    D_MERGER{"community_merger\nprovided?"}
    MERGER["QueryGuidedCommunityMerger\ncontext-aware community merge"]

    subgraph HOP_LOOP["Hop Loop  1 … max_hop"]
        direction TB

        EXPAND["Adapter.get_neighbors()\nmax_neighbors cap\ncontext_embedding ranking"]

        D_TEMPORAL{"temporal filter\nhard or soft?"}
        TEMP_FILT["is_valid_at()\nvalid_from / valid_to\nhard prune or decay only"]

        D_CVT{"cvt_passthrough\n+ is CVT node?"}
        CVT["collapse A→CVT→B\nsingle hop\nCVT_HOP_PENALTY × 0.85"]

        D_SYMVAL{"SymbolicValidator\nattached?"}
        SYMVAL["validate_step()\nlogic guardrail\nreject invalid steps"]

        CSA_WEIGHT["CSAEngine.compute_weight_with_features()\nReasoningLogit\n10-signal vector"]

        subgraph CSA_FORMULA["CSA Attention Formula"]
            direction LR
            F1["α · sim\nsemantic cosine"]
            F2["β · cs\ncommunity score"]
            F3["γ · etw\nedge-type weight"]
            F4["−δ · nd\ndistance penalty"]
            F5["ε · hd\nhop decay"]
            F6["ζ · pr_v\nPageRank prior"]
            F7["η · td\ntemporal decay"]
            F8["ι · nr_v\nnode recency"]
            F9["−μ · sd\nsynth density penalty"]
            F10["θ · grounding\nconfidence score"]
        end

        LOGIT_SCORE["ReasoningLogit.score(params)\nsigmoid(sum)\nweight ∈ 0..1"]

        EMB_AGG["embedding aggregation\nReLU(w·v_emb + h) + h\nLayerNorm residual"]

        D_CALENG{"CalibrationEngine\nattached?"}
        CAL["calibrate_hop(weights, tails)\nentropy check\nuncertainty_log\nconfidence multiplier"]

        D_BRIDGE{"BridgeTwinEngine\nattached?"}
        BRIDGE_REC["record_crossing()\nBridgeTwin synthesis\nINSIGHT_LINK edges"]

        D_PRUNE_MODE{"traversal type"}
        PRUNE_STD["sort by path.score\nnlargest(beam_width)"]
        PRUNE_PROB["Thompson sample\nBeta(α,β)\nnlargest(beam_width)"]
        PRUNE_ENGRAM["_boosted_score()\naffinity × engram_strength\nnlargest(beam_width)"]
    end

    ALL_PATHS["all_paths collected\nacross all hops"]
end

%% ═══════════════════════════════════════════════════════════
%% ANSWER EXTRACTION
%% ═══════════════════════════════════════════════════════════

subgraph EXTRACT["⑤ Answer Extraction"]
    direction TB

    PATH_SCORE["PathScorer\nattention · community · semantic\ngrounding · relation_prior\nweighted sum → path_score"]

    D_PRIOR{"RelationPathPrior\nor GraphRelationPrior?"}
    PRIOR["prior.score(path)\nboost known relation chains"]

    VOTE["convergence voting\nvote_weight × normalised_count\n+ (1−vote_weight) × path_score"]

    DEDUP["entity deduplication\nbest path per entity"]

    TOPK["top_k Answer objects\nentity_id · score · best_path\nedge_features · community_sequence\npath_confidence"]
end

%% ═══════════════════════════════════════════════════════════
%% OUTPUT LAYER
%% ═══════════════════════════════════════════════════════════

subgraph OUTPUT["⑥ Output & Verbalization"]
    direction LR

    ENGRAM_VERB["EngramVerbalizer\ncompressed trace\nEngram:[Newton~>Leibniz!>Calculus]"]
    PATH_VERB["PathVerbalizer\nfluent NL sentences\nwith edge citations"]
    JSON_OUT["Structured JSON\nQueryResponse\nranked answers + paths\npartial: bool  error: str"]

    D_FAULT{"traversal\nexception?"}
    PARTIAL_OUT["partial=True\nerror=str(exc)\n_partial_paths used\nHTTP 200"]

    D_OUT_TYPE{"caller type"}
    REST_OUT["REST /query response\nQueryResponse schema"]
    UI_OUT["Gradio HTML\nEngram trace + path cards\nattn radar plot"]
    CLI_OUT["CLI stdout\nranked entities + scores"]
    FED_OUT["TraversalBranchResponse\nfor parent node merge"]
    STREAM_OUT["NDJSON stream\nhop-by-hop paths\n+ terminal error chunk on fault"]
end

%% ═══════════════════════════════════════════════════════════
%% LEARNING LOOPS
%% ═══════════════════════════════════════════════════════════

subgraph LEARNING["⑦ Adaptive Learning Loops"]
    direction TB

    subgraph ONLINE["Online — per query"]
        META_L["MetaParameterLearner\nSGD on community params\nmomentum update\nper POST /feedback"]
    end

    subgraph BATCH["Batch — explicit trigger"]
        CSA_L["CSAParameterLearner\npairwise margin loss\ngradient descent\nPOST /retrain"]
    end

    subgraph TEMPORAL_CAL["Temporal — grid search"]
        T_CAL["TemporalCalibrator\neta × iota grid\nRecall@K objective\napply() writes to CSAEngine"]
    end

    subgraph ENGRAM_LEARN["Pattern — per result"]
        ENGRAM_REC["EngramTraversal\n.record_answers()\nrelation sequence → Engram\naffinity index rebuilt"]
        ENGRAM_PERSIST["Engram.save(path)\nJSON serialization\nsurvives restart"]
    end

    subgraph QLOG["Query Log — durability"]
        QLOG_W["QueryLog.record()\nappend NDJSON\nseeds + answers + rel_seqs"]
        QLOG_REPLAY["QueryLog.replay_into_cache()\nwarm Engram on startup"]
    end
end

%% ═══════════════════════════════════════════════════════════
%% BACKGROUND SYSTEMS
%% ═══════════════════════════════════════════════════════════

subgraph BACKGROUND["⑧ Background Systems"]
    direction TB

    subgraph REM["REM Engine — self-reorganization"]
        REM_PRUNE["_prune()\nconfidence < threshold\ndead edges removed"]
        REM_CONS["_consolidate()\nredundant paths merged\nDSCF recompute"]
        REM_SYNTH["_synthesize()\nSynaptic Bridge bridge synthesis\nsd penalty applied"]
        REM_SCHED["schedule(interval_s)\ndaemon timer"]
    end

    subgraph INSIGHT["Insight Stack"]
        INS_HOT["InsightEngine hot path\nO(1) ring buffer append\nevery cross-community hop"]
        INS_WARM["warm path daemon\nsurprise score\nInsightEvent fired\nHebbian boost"]
        INS_COLD["cold path scan\nscan_boundaries()\nbatch cosine sim\nmat_a @ mat_b.T\nLRU baseline eviction"]
        INS_MAT["materialize()\nINSIGHT_LINK edge added\nweight=2.0  conf=0.85"]
        INS_VAL["InsightValidator\nbilateral reverse BFS\ncorroboration check\nconfidence promotion"]
        INS_META["MetaInsightEngine\nsecond-order patterns\nshared-entity clusters\ntemporal proximity"]
    end

    subgraph HYPO["Hypothesis Engine"]
        HYPO_GEN["HypothesisEngine.generate()\ncontradiction-driven\nabductive multi-path\nPOST /hypothesize"]
        HYPO_MAT["materialize()\nsynth edges ingested"]
        EXT_VAL["ExternalValidator\nresearch_agent scan\nPOST /research/validate"]
    end

    subgraph REBAL["Global Rebalancer"]
        REBAL_MON["GlobalRebalancer\nmodularity Q drift\nmonitor(interval_s)"]
        REBAL_DSCF["background full DSCF re-run\nnew community_map\nbridge_engine pruned"]
    end

    subgraph STREAM_SYS["Stream Adapter"]
        ST_SIM["Simulated source"]
        ST_TAIL["File Tail source"]
        ST_HTTP["HTTP Polling source"]
        ST_PROC["StreamAdapter\nPythonCallbackSource\nring buffer events\nmutation log"]
    end
end

%% ═══════════════════════════════════════════════════════════
%% PERSISTENCE
%% ═══════════════════════════════════════════════════════════

subgraph PERSIST["⑨ Persistence Layer"]
    direction LR
    P_SAVE["save_state()\npickle adapter + embeddings\ncommunity_map + CSA metadata"]
    P_LOAD["load_state()\nrestore full session\n+ rebuild REM / Insight engines"]
    P_CACHE["build caches\nembeddings.pkl\nembeddings_sage.pkl\ncommunities.pkl"]
    P_ENGRAM["Engram JSON\nrelation pattern store"]
    P_QLOG["QueryLog NDJSON\nquery history\nEngram warm-up source"]
    P_PROV["ProvenanceLedger\nper-batch/cycle edge records\nrollback_batch() / rollback_cycle()"]
    P_SNAP["GraphSnapshot JSON\nportable topology\nsave() / restore() / diff()"]
end

%% ═══════════════════════════════════════════════════════════
%% API SURFACE
%% ═══════════════════════════════════════════════════════════

subgraph API_SURF["⑩ API Surface"]
    direction LR
    API1["GET  /health\nPOST /query\nGET  /query/stream"]
    API2["POST /feedback\nPOST /retrain\nGET  /params\nPOST /params"]
    API3["GET  /communities\nGET  /graph/edges\nGET  /bridges\nGET  /logs\nDEL  /logs"]
    API4["POST /traverse\nfederated branch"]
    API5["POST /hypothesize\nPOST /hypothesize/materialize\nPOST /hypothesize/rollback"]
    API6["POST /research/start\nPOST /research/scan\nPOST /research/validate\nGET  /research/proposals\nGET/POST /research/auto-approver"]
    API10["POST /research/loop/start\nPOST /research/loop/stop\nGET  /research/loop/status\nPOST /research/loop/configure"]
    API11["GET  /research/provenance/stats\nGET  /research/provenance/batches\nPOST /research/provenance/rollback/{id}\nPOST /research/provenance/rollback-cycle/{n}"]
    API7["POST /insight/validate/all\nPOST /insight/validate/id"]
    API8["POST /stream/ingest\nGET  /stream/status\nGET  /stream/events"]
    API9["POST /build\nCSV hot-reload"]
end

%% ═══════════════════════════════════════════════════════════
%% MAIN FLOW EDGES
%% ═══════════════════════════════════════════════════════════

E_REST --> SEED_FIND
E_CLI  --> SEED_FIND
E_UI   --> SEED_FIND
E_FED  --> SEED_FIND
E_STREAM --> ST_SIM & ST_TAIL & ST_HTTP

ADAPTERS --> D_PIPE
D_PIPE -->|yes| PIPE --> D_STDP
D_PIPE -->|no| D_STDP
D_STDP -->|yes| STDP --> D_SIG
D_STDP -->|no| D_SIG
D_SIG  -->|yes| SIG --> D_COMPLETE
D_SIG  -->|no| D_COMPLETE
D_COMPLETE -->|yes| COMPLETE --> D_ENHANCE
D_COMPLETE -->|no| D_ENHANCE
D_ENHANCE  -->|yes| ENHANCE --> NX_ADAPTER
D_ENHANCE  -->|no| NX_ADAPTER

NX_ADAPTER --> CACHE_CHK

CACHE_CHK -->|hit| CSA_BUILD
CACHE_CHK -->|miss| D_EMB

D_EMB -->|random| EMB_RAND --> EMB_SAVE
D_EMB -->|sentence| EMB_SENT --> EMB_SAVE
EMB_SAVE --> D_SAGE

D_SAGE -->|yes| SAGE_CACHE
SAGE_CACHE -->|hit| STRUCT
SAGE_CACHE -->|miss| SAGE --> STRUCT
D_SAGE -->|no| STRUCT

STRUCT --> D_COMM
D_COMM -->|dscf| C_DSCF --> D_COARSEN
D_COMM -->|tsc| C_TSC --> D_COARSEN
D_COMM -->|leiden| C_LEIDEN --> D_COARSEN
D_COMM -->|lpa| C_LPA --> D_COARSEN
D_COARSEN -->|yes| COARSEN --> COMM_SAVE
D_COARSEN -->|no| COMM_SAVE
COMM_SAVE --> CSA_BUILD

CSA_BUILD --> D_PROB
D_PROB -->|yes| BT_PROB
D_PROB -->|no| D_ENGRAM_CACHE
D_ENGRAM_CACHE -->|yes| BT_ENGRAM
D_ENGRAM_CACHE -->|no| BT_STD

BT_STD & BT_PROB & BT_ENGRAM --> SEED_FIND

SEED_FIND --> D_QUERY_EMB
D_QUERY_EMB -->|yes| Q_EMB --> Q_SNAPSHOT
D_QUERY_EMB -->|no| Q_SNAPSHOT

Q_SNAPSHOT --> D_MERGER
D_MERGER -->|yes| MERGER --> EXPAND
D_MERGER -->|no| EXPAND

EXPAND --> D_TEMPORAL
D_TEMPORAL -->|hard| TEMP_FILT
D_TEMPORAL -->|soft decay| TEMP_FILT
TEMP_FILT --> D_CVT

D_CVT -->|yes + CVT node| CVT --> D_SYMVAL
D_CVT -->|no| D_SYMVAL

D_SYMVAL -->|yes| SYMVAL -->|reject| EXPAND
SYMVAL -->|accept| CSA_WEIGHT
D_SYMVAL -->|no| CSA_WEIGHT

CSA_WEIGHT --> F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 & F9 & F10
F1 & F2 & F3 & F4 & F5 & F6 & F7 & F8 & F9 & F10 --> LOGIT_SCORE

LOGIT_SCORE --> EMB_AGG

D_BRIDGE -->|yes| BRIDGE_REC
EMB_AGG --> D_BRIDGE
D_BRIDGE -->|no| D_CALENG

BRIDGE_REC --> D_CALENG

D_CALENG -->|yes| CAL --> D_PRUNE_MODE
D_CALENG -->|no| D_PRUNE_MODE

D_PRUNE_MODE -->|standard| PRUNE_STD --> ALL_PATHS
D_PRUNE_MODE -->|probabilistic| PRUNE_PROB --> ALL_PATHS
D_PRUNE_MODE -->|Engram| PRUNE_ENGRAM --> ALL_PATHS

ALL_PATHS --> PATH_SCORE
PATH_SCORE --> D_PRIOR
D_PRIOR -->|yes| PRIOR --> VOTE
D_PRIOR -->|no| VOTE
VOTE --> DEDUP --> TOPK

TOPK --> ENGRAM_VERB & PATH_VERB & JSON_OUT
JSON_OUT --> D_OUT_TYPE
D_OUT_TYPE -->|REST| REST_OUT
D_OUT_TYPE -->|UI| UI_OUT
D_OUT_TYPE -->|CLI| CLI_OUT
D_OUT_TYPE -->|federated| FED_OUT
D_OUT_TYPE -->|stream| STREAM_OUT

TOPK --> ENGRAM_REC --> ENGRAM_PERSIST
TOPK --> QLOG_W

QLOG_REPLAY -.->|startup warm-up| ENGRAM_REC
P_ENGRAM -.->|load on init| ENGRAM_REC

META_L -.->|updates| CSA_BUILD
CSA_L  -.->|updates| CSA_BUILD
T_CAL  -.->|writes eta/iota| CSA_BUILD

INS_HOT --> INS_WARM --> INS_MAT
INS_COLD --> INS_MAT
INS_MAT --> INS_VAL --> INS_META

REBAL_MON --> REBAL_DSCF -.->|new partition| CSA_BUILD

REM_SCHED --> REM_PRUNE --> REM_CONS --> REM_SYNTH
REM_SYNTH -.->|modified graph| NX_ADAPTER

HYPO_GEN --> HYPO_MAT -.->|synth edges| NX_ADAPTER
HYPO_MAT --> EXT_VAL

subgraph EVO["⑤ EVOLUTION — Self-Refining Program"]
    direction TB
    DMN["DefaultModeEngine\nHEURISTIC_BOTTLENECK\n(audit idle cycles)"]
    RESEARCHER["AutonomousResearcher\n(Recursive Self-Synthesis)"]
    SYNTH_MOD["synthesize_module()\n(Code Generation)"]
    
    DMN -->|"insights"| RESEARCHER
    RESEARCHER -->|"propose"| SYNTH_MOD
    SYNTH_MOD -.->|"hot-patch"| CORTEX
end

ST_SIM & ST_TAIL & ST_HTTP --> ST_PROC -.->|ingest events| NX_ADAPTER

P_SAVE -.->|snapshot| NX_ADAPTER
P_LOAD -.->|restore| NX_ADAPTER
P_CACHE -.->|accelerate| CACHE_CHK & SAGE_CACHE

API1 & API2 & API3 & API4 & API5 & API6 & API7 & API8 & API9 & API10 & API11 -.->|routes to| CORTEX

%% ═══════════════════════════════════════════════════════════
%% STYLE
%% ═══════════════════════════════════════════════════════════

classDef thalamus   fill:#1a3a2a,stroke:#2d6a4f,color:#d8f3dc
classDef cortex     fill:#1a2a3a,stroke:#2d4a6f,color:#bde0fe
classDef learning   fill:#2a1a3a,stroke:#6a2d8f,color:#e9c7ff
classDef background fill:#3a2a1a,stroke:#8f6a2d,color:#ffe0b0
classDef persist    fill:#1a3a3a,stroke:#2d7a7a,color:#b0f0f0
classDef output     fill:#2a3a1a,stroke:#4a7a2d,color:#d0f0b0
classDef entry      fill:#3a1a1a,stroke:#8f2d2d,color:#ffb0b0
classDef api        fill:#1a1a1a,stroke:#555,color:#ccc
classDef decision   fill:#2a2a1a,stroke:#8f8f2d,color:#f0f0b0

class ADAPTERS,PIPE,STDP,SIG,COMPLETE,ENHANCE,NX_ADAPTER thalamus
class EMB_RAND,EMB_SENT,EMB_SAVE,SAGE,STRUCT,C_DSCF,C_TSC,C_LEIDEN,C_LPA,COARSEN,COMM_SAVE,CSA_BUILD,BT_STD,BT_PROB,BT_ENGRAM thalamus
class SEED_FIND,Q_EMB,Q_SNAPSHOT,MERGER,EXPAND,TEMP_FILT,CVT,SYMVAL,CSA_WEIGHT,LOGIT_SCORE,EMB_AGG,CAL,BRIDGE_REC,PRUNE_STD,PRUNE_PROB,PRUNE_ENGRAM,ALL_PATHS cortex
class F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,PATH_SCORE,PRIOR,VOTE,DEDUP,TOPK cortex
class META_L,CSA_L,T_CAL,ENGRAM_REC,ENGRAM_PERSIST,QLOG_W,QLOG_REPLAY learning
class REM_PRUNE,REM_CONS,REM_SYNTH,REM_SCHED,INS_HOT,INS_WARM,INS_COLD,INS_MAT,INS_VAL,INS_META,HYPO_GEN,HYPO_MAT,EXT_VAL,REBAL_MON,REBAL_DSCF,ST_SIM,ST_TAIL,ST_HTTP,ST_PROC background
class P_SAVE,P_LOAD,P_CACHE,P_ENGRAM,P_QLOG persist
class ENGRAM_VERB,PATH_VERB,JSON_OUT,REST_OUT,UI_OUT,CLI_OUT,FED_OUT,STREAM_OUT output
class E_REST,E_CLI,E_UI,E_FED,E_STREAM entry
class API1,API2,API3,API4,API5,API6,API7,API8,API9,API10,API11 api
class P_PROV,P_SNAP persist
class DMN,RESEARCHER,SYNTH_MOD learning
class D_PIPE,D_STDP,D_SIG,D_COMPLETE,D_ENHANCE,D_EMB,D_SAGE,D_COMM,D_COARSEN,D_PROB,D_ENGRAM_CACHE,D_QUERY_EMB,D_MERGER,D_TEMPORAL,D_CVT,D_SYMVAL,D_BRIDGE,D_CALENG,D_PRUNE_MODE,D_PRIOR,D_OUT_TYPE,CACHE_CHK,SAGE_CACHE decision
```

---

## Legend

| Colour | Layer | Description |
|---|---|---|
| Dark green | THALAMUS | Ingestion, embedding, community detection |
| Dark blue | CORTEX | Traversal, attention scoring, answer extraction |
| Purple | Learning | Online SGD, batch retrain, temporal calibration, Engram pattern cache |
| Orange | Background | REM, InsightEngine, HypothesisEngine, Rebalancer, Stream |
| Teal | Persistence | State snapshots, QueryLog, Engram JSON, build caches |
| Olive | Output | Verbalization, response routing |
| Red | Entry | REST, CLI, UI, Federated, Stream |
| Grey | API | All REST endpoint groups |
| Yellow | Decision | Every branching/option node |
| Cyan | Visualization | TelemetryBridge, UE5 client, WebSocket stream |

## Key Decision Points

| Decision | Options | Effect |
|---|---|---|
| IngestionPipeline | on / off | entity normalization, dedup, relation canonicalization |
| STDPDiscretizer | on / off | causal edge inference from spike timing |
| SignalEncoder | on / off | cross-modal sensor → embedding space alignment |
| Embedding mode | random / sentence | random=fast+test; sentence=BGE-Small-v1.5 384-dim |
| GraphSAGE | on / off | neighbourhood smoothing enriches semantic (α) signal |
| Community engine | DSCF / TSC / Leiden / LPA | affects attention head structure |
| Coarsening | min_size / target_max / none | merges small communities |
| Traversal mode | standard / probabilistic / Engram | changes beam pruning strategy |
| Temporal filter | hard prune / soft decay | edges outside window rejected or penalised |
| CVT passthrough | on / off | Freebase mediator collapse for WebQSP |
| SymbolicValidator | on / off | per-step logical guardrail |
| CalibrationEngine | on / off | self-doubt entropy check per hop |
| RelationPathPrior | on / off | boosts known relation chain patterns |
| Engram | warm / cold | steers beam toward cached relation sequences |
| LoopedBeamTraversal | max_loops=1..N | applies traversal T times with inter-loop semantic/metabolic/mnemonic feedback |
| AutoApprover | attached / not | tiered auto-decision on ResearchFindings; online SGD from confirmations |
| ProvenanceLedger | attached / not | per-batch edge recording; enables targeted rollback |
| GraphSnapshot | save / restore / diff | portable JSON topology persistence across restarts |
| Adaptive Loop Tuning | on / off | DiscoveryCalibrator-driven dynamic cap and interval scaling per cycle |
| TelemetryBridge | ws_port set / not set | when set, starts WebSocket server; /query emits SYNAPTIC_PULSE, /research/approve emits SYNAPTOGENESIS, /rem/run emits SYNAPTIC_PRUNE |
| UE5 Visualization | bPreferLayoutFile true / false | true: loads pre-computed graph_layout.json (exact Fibonacci positions); false: derives positions at runtime via REST |
| Active Inference | active_inference on / off | when on, AutonomousDiscoveryLoop seeds idle-period queries from high-PE nodes or random selection to surface weak priors |
| GUI Adaptation | gui_adaptation on / off | when on, GUIAdaptationEngine watches metabolic + loop signals and modifies UE5 HUD structurally (via ue-llm-toolkit) and at runtime (via GUI_ADAPTATION WebSocket events) |

# #   P h a s e   1 1 0 :   G l o b a l   W o r k s p a c e   ( G W S )  
 I m p l e m e n t s   a   b l a c k b o a r d - b a s e d   c o m p e t i t i v e   a t t e n t i o n   l a y e r ,   r e p l a c i n g   l i n e a r   M A C H   e s c a l a t i o n   w i t h   d y n a m i c   s i g n a l   b i d d i n g .  
 # #   P h a s e   1 1 0 :   G l o b a l   W o r k s p a c e   ( G W S ) 
 I m p l e m e n t s   a   b l a c k b o a r d - b a s e d   c o m p e t i t i v e   a t t e n t i o n   l a y e r ,   r e p l a c i n g   l i n e a r   M A C H   e s c a l a t i o n   w i t h   d y n a m i c   s i g n a l   b i d d i n g .   C o m m u n i t i e s   b r o a d c a s t   ' s u r p r i s e '   s i g n a l s   t o   a   s h a r e d   B l a c k b o a r d ,   a n d   t h e   C o n s e n s u s H i e r a r c h y E n g i n e   d y n a m i c a l l y   b o o s t s   c a n d i d a t e s   w i t h   h i g h - n o v e l t y   e v i d e n c e   b e f o r e   s t a n d a r d   e s c a l a t i o n   o c c u r s .  
 # #   P h a s e   1 1 1 :   A c t i v e   I n f e r e n c e 
 T r a n s f o r m s   r e a s o n i n g   f r o m   a   r e a c t i v e   s e a r c h   t o   a   p r o a c t i v e   t r a v e r s a l .   T h e   P r e d i c t i v e C o d e r   g e n e r a t e s   ' E x p e c t e d   P a t h '   p r i o r s   f r o m   E n g r a m   p a t t e r n s ,   w h i c h   b i a s   t h e   B e a m T r a v e r s a l   t o w a r d   l i k e l y   s e q u e n c e s .   P r e d i c t i o n   E r r o r s   ( P E )   t r i g g e r   m e t a b o l i c   a r o u s a l ,   a l l o w i n g   t h e   s y s t e m   t o   f o c u s   c o m p u t a t i o n a l   e n e r g y   o n   ' s u r p r i s i n g '   d i s c o v e r i e s .  
 