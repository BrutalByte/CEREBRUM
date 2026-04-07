# CEREBRUM System Architecture

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

    D_AAAK_CACHE{"AAAKCache\npath provided?"}
    BT_AAAK["AAAKBeamTraversal\n_prune_candidates override\naffinity-boosted scoring\nrelation-pattern steering"]
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
        PRUNE_AAAK["_boosted_score()\naffinity × aaak_strength\nnlargest(beam_width)"]
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

    AAAK_VERB["AAAKVerbalizer\ncompressed trace\nAAK:[Newton~>Leibniz!>Calculus]"]
    PATH_VERB["PathVerbalizer\nfluent NL sentences\nwith edge citations"]
    JSON_OUT["Structured JSON\nQueryResponse\nranked answers + paths"]

    D_OUT_TYPE{"caller type"}
    REST_OUT["REST /query response\nQueryResponse schema"]
    UI_OUT["Gradio HTML\nAAK trace + path cards\nattn radar plot"]
    CLI_OUT["CLI stdout\nranked entities + scores"]
    FED_OUT["TraversalBranchResponse\nfor parent node merge"]
    STREAM_OUT["NDJSON stream\nhop-by-hop paths"]
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

    subgraph AAAK_LEARN["Pattern — per result"]
        AAAK_REC["AAAKBeamTraversal\n.record_answers()\nrelation sequence → AAAKCache\naffinity index rebuilt"]
        AAAK_PERSIST["AAAKCache.save(path)\nJSON serialization\nsurvives restart"]
    end

    subgraph QLOG["Query Log — durability"]
        QLOG_W["QueryLog.record()\nappend NDJSON\nseeds + answers + rel_seqs"]
        QLOG_REPLAY["QueryLog.replay_into_cache()\nwarm AAAKCache on startup"]
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
        REM_SYNTH["_synthesize()\nwormhole bridge synthesis\nsd penalty applied"]
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
    P_AAAK["AAAKCache JSON\nrelation pattern store"]
    P_QLOG["QueryLog NDJSON\nquery history\nAAK warm-up source"]
end

%% ═══════════════════════════════════════════════════════════
%% API SURFACE
%% ═══════════════════════════════════════════════════════════

subgraph API_SURF["⑩ API Surface"]
    direction LR
    API1["GET  /health\nPOST /query\nGET  /query/stream"]
    API2["POST /feedback\nPOST /retrain\nGET  /params\nPOST /params"]
    API3["GET  /communities\nGET  /bridges\nGET  /logs\nDEL  /logs"]
    API4["POST /traverse\nfederated branch"]
    API5["POST /hypothesize\nPOST /hypothesize/materialize\nPOST /hypothesize/rollback"]
    API6["POST /research/start\nPOST /research/scan\nPOST /research/validate\nGET  /research/proposals"]
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
D_PROB -->|no| D_AAAK_CACHE
D_AAAK_CACHE -->|yes| BT_AAAK
D_AAAK_CACHE -->|no| BT_STD

BT_STD & BT_PROB & BT_AAAK --> SEED_FIND

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
D_PRUNE_MODE -->|AAAK| PRUNE_AAAK --> ALL_PATHS

ALL_PATHS --> PATH_SCORE
PATH_SCORE --> D_PRIOR
D_PRIOR -->|yes| PRIOR --> VOTE
D_PRIOR -->|no| VOTE
VOTE --> DEDUP --> TOPK

TOPK --> AAAK_VERB & PATH_VERB & JSON_OUT
JSON_OUT --> D_OUT_TYPE
D_OUT_TYPE -->|REST| REST_OUT
D_OUT_TYPE -->|UI| UI_OUT
D_OUT_TYPE -->|CLI| CLI_OUT
D_OUT_TYPE -->|federated| FED_OUT
D_OUT_TYPE -->|stream| STREAM_OUT

TOPK --> AAAK_REC --> AAAK_PERSIST
TOPK --> QLOG_W

QLOG_REPLAY -.->|startup warm-up| AAAK_REC
P_AAAK -.->|load on init| AAAK_REC

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

ST_SIM & ST_TAIL & ST_HTTP --> ST_PROC -.->|ingest events| NX_ADAPTER

P_SAVE -.->|snapshot| NX_ADAPTER
P_LOAD -.->|restore| NX_ADAPTER
P_CACHE -.->|accelerate| CACHE_CHK & SAGE_CACHE

API1 & API2 & API3 & API4 & API5 & API6 & API7 & API8 & API9 -.->|routes to| CORTEX

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
class EMB_RAND,EMB_SENT,EMB_SAVE,SAGE,STRUCT,C_DSCF,C_TSC,C_LEIDEN,C_LPA,COARSEN,COMM_SAVE,CSA_BUILD,BT_STD,BT_PROB,BT_AAAK thalamus
class SEED_FIND,Q_EMB,Q_SNAPSHOT,MERGER,EXPAND,TEMP_FILT,CVT,SYMVAL,CSA_WEIGHT,LOGIT_SCORE,EMB_AGG,CAL,BRIDGE_REC,PRUNE_STD,PRUNE_PROB,PRUNE_AAAK,ALL_PATHS cortex
class F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,PATH_SCORE,PRIOR,VOTE,DEDUP,TOPK cortex
class META_L,CSA_L,T_CAL,AAAK_REC,AAAK_PERSIST,QLOG_W,QLOG_REPLAY learning
class REM_PRUNE,REM_CONS,REM_SYNTH,REM_SCHED,INS_HOT,INS_WARM,INS_COLD,INS_MAT,INS_VAL,INS_META,HYPO_GEN,HYPO_MAT,EXT_VAL,REBAL_MON,REBAL_DSCF,ST_SIM,ST_TAIL,ST_HTTP,ST_PROC background
class P_SAVE,P_LOAD,P_CACHE,P_AAAK,P_QLOG persist
class AAAK_VERB,PATH_VERB,JSON_OUT,REST_OUT,UI_OUT,CLI_OUT,FED_OUT,STREAM_OUT output
class E_REST,E_CLI,E_UI,E_FED,E_STREAM entry
class API1,API2,API3,API4,API5,API6,API7,API8,API9 api
class D_PIPE,D_STDP,D_SIG,D_COMPLETE,D_ENHANCE,D_EMB,D_SAGE,D_COMM,D_COARSEN,D_PROB,D_AAAK_CACHE,D_QUERY_EMB,D_MERGER,D_TEMPORAL,D_CVT,D_SYMVAL,D_BRIDGE,D_CALENG,D_PRUNE_MODE,D_PRIOR,D_OUT_TYPE,CACHE_CHK,SAGE_CACHE decision
```

---

## Legend

| Colour | Layer | Description |
|---|---|---|
| Dark green | THALAMUS | Ingestion, embedding, community detection |
| Dark blue | CORTEX | Traversal, attention scoring, answer extraction |
| Purple | Learning | Online SGD, batch retrain, temporal calibration, AAAK pattern cache |
| Orange | Background | REM, InsightEngine, HypothesisEngine, Rebalancer, Stream |
| Teal | Persistence | State snapshots, QueryLog, AAAK JSON, build caches |
| Olive | Output | Verbalization, response routing |
| Red | Entry | REST, CLI, UI, Federated, Stream |
| Grey | API | All REST endpoint groups |
| Yellow | Decision | Every branching/option node |

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
| Traversal mode | standard / probabilistic / AAAK | changes beam pruning strategy |
| Temporal filter | hard prune / soft decay | edges outside window rejected or penalised |
| CVT passthrough | on / off | Freebase mediator collapse for WebQSP |
| SymbolicValidator | on / off | per-step logical guardrail |
| CalibrationEngine | on / off | self-doubt entropy check per hop |
| RelationPathPrior | on / off | boosts known relation chain patterns |
| AAAKCache | warm / cold | steers beam toward cached relation sequences |
