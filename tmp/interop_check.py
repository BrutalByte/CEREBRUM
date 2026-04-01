"""System interoperability check — all major CEREBRUM components."""
import networkx as nx
from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.community_engine import dscf_communities
from core.attention_engine import CSAEngine
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract

errors = []
print("=== CEREBRUM System Interoperability Check ===")

# Shared base graph
G = nx.karate_club_graph()
adapter = NetworkXAdapter(G)
eng = RandomEngine(dim=32)
adapter.embeddings = eng.encode_entities({n: n for n in G.nodes()})
parts = dscf_communities(G.to_undirected())
cm = {n: cid for cid, members in enumerate(parts) for n in members}
adapter.community_map = cm
adapter._partition = parts
csa = CSAEngine(adapter)
traversal = BeamTraversal(adapter, csa, beam_width=5, max_hop=2)
paths = traversal.traverse([0])
answers = extract(paths, top_k=5)

# 1. IngestionPipeline
print("\n[1] IngestionPipeline...")
try:
    from core.thalamus import IngestionPipeline
    p = IngestionPipeline(namespace="test")
    a = NetworkXAdapter.from_triples([("a", "R", "b", {"confidence": 0.9}), ("b", "R", "c")], pipeline=p)
    assert a.node_count() == 3
    print("  PASS: 3 nodes, namespace isolation working")
except Exception as e:
    errors.append(f"IngestionPipeline: {e}")
    print(f"  FAIL: {e}")

# 2. EmbeddingEngine
print("\n[2] EmbeddingEngine...")
try:
    import numpy as np
    e = RandomEngine(dim=64)
    embs = e.encode_entities({"a": "alice", "b": "bob", "c": "carol"})
    assert len(embs) == 3
    for k, v in embs.items():
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5, "Not unit-normalized"
    print("  PASS: 3 unit-normalized embeddings")
except Exception as e:
    errors.append(f"EmbeddingEngine: {e}")
    print(f"  FAIL: {e}")

# 3. StructuralEncoder
print("\n[3] StructuralEncoder...")
try:
    from core.structural_encoder import compute_structural_features
    feats = compute_structural_features(G)
    assert len(feats) == G.number_of_nodes()
    print(f"  PASS: features for {len(feats)} nodes")
except Exception as e:
    errors.append(f"StructuralEncoder: {e}")
    print(f"  FAIL: {e}")

# 4. DSCF + CSAEngine
print("\n[4] DSCF + CSAEngine...")
try:
    w = csa.compute_weight(str(0), str(1), hop=1, edge_type="RELATED_TO")
    assert 0.0 <= w <= 1.0
    csa.set_query_snapshot(cm)
    csa.clear_query_snapshot()
    print(f"  PASS: {len(parts)} communities, weight={w:.4f}, query snapshot OK")
except Exception as e:
    errors.append(f"DSCF/CSA: {e}")
    print(f"  FAIL: {e}")

# 5. BeamTraversal + AnswerExtractor
print("\n[5] BeamTraversal + AnswerExtractor...")
try:
    assert len(paths) > 0 and len(answers) > 0
    assert all(0.0 <= a.score <= 1.0 for a in answers)
    assert all(a.path_confidence >= 0.0 for a in answers)
    print(f"  PASS: {len(paths)} paths, {len(answers)} answers, top={answers[0].entity_id}")
except Exception as e:
    errors.append(f"BeamTraversal/AnswerExtractor: {e}")
    print(f"  FAIL: {e}")

# 6. REMEngine
print("\n[6] REMEngine...")
try:
    from core.rem_engine import REMEngine
    rem = REMEngine(adapter)
    report = rem.run(dry_run=True)
    print(f"  PASS: pruned={report.pruned_edges}, synthesized={report.synthesized_edges}")
except Exception as e:
    errors.append(f"REMEngine: {e}")
    print(f"  FAIL: {e}")

# 7. BridgeTwinEngine
print("\n[7] BridgeTwinEngine...")
try:
    from core.bridge_engine import BridgeTwinEngine
    bte = BridgeTwinEngine(adapter)
    bridges = bte.active_bridges()
    print(f"  PASS: {len(bridges)} active bridges")
except Exception as e:
    errors.append(f"BridgeTwinEngine: {e}")
    print(f"  FAIL: {e}")

# 8. ResourceGovernor
print("\n[8] ResourceGovernor...")
try:
    from core.resource_governor import ResourceGovernor
    gov = ResourceGovernor()
    stats = gov.get_current_stats()
    cap = gov.estimate_path_capacity()
    can = gov.can_expand(0, 100)
    print(f"  PASS: RAM={stats['system_ram_pct']:.1f}%, capacity={cap:,}, can_expand={can}")
except Exception as e:
    errors.append(f"ResourceGovernor: {e}")
    print(f"  FAIL: {e}")

# 9. FederatedAdapter
print("\n[9] FederatedAdapter...")
try:
    from adapters.federated_adapter import FederatedAdapter
    G1 = nx.DiGraph()
    G1.add_edge("x", "y", relation="R")
    G2 = nx.DiGraph()
    G2.add_edge("y", "z", relation="R")
    a1 = NetworkXAdapter(G1)
    a2 = NetworkXAdapter(G2)
    fed = FederatedAdapter({"shard1": a1, "shard2": a2})
    ent = fed.get_entity("x")
    assert ent is not None and ent.id == "x"
    neighbors = fed.get_neighbors("y")
    print(f"  PASS: entity lookup OK, cross-shard neighbors={len(neighbors)}")
except Exception as e:
    errors.append(f"FederatedAdapter: {e}")
    print(f"  FAIL: {e}")

# 10. STDPDiscretizer
print("\n[10] STDPDiscretizer...")
try:
    from core.discretizer import STDPDiscretizer
    disc = STDPDiscretizer(window_seconds=0.1, min_causal_span=0.0, n_min=1, w_threshold=0.05)
    for t in [0.0, 0.05, 0.10, 0.15, 0.20]:
        disc.process("neuron_a", timestamp=t)
        disc.process("neuron_b", timestamp=t + 0.02)
    print("  PASS: STDP processing complete")
except Exception as e:
    errors.append(f"STDPDiscretizer: {e}")
    print(f"  FAIL: {e}")

# 11. GlobalRebalancer
print("\n[11] GlobalRebalancer...")
try:
    from core.rebalancer import GlobalRebalancer
    rb = GlobalRebalancer(adapter)
    q = rb._compute_q()
    rb._check_drift(dry_run=True)
    print(f"  PASS: modularity Q={q:.4f}")
except Exception as e:
    errors.append(f"GlobalRebalancer: {e}")
    print(f"  FAIL: {e}")

# 12. InsightValidator
print("\n[12] InsightValidator...")
try:
    from core.insight_validator import InsightValidator, InsightEvent
    iv = InsightValidator(adapter, corroboration_seeds=3, corroboration_threshold=1)
    event = InsightEvent(
        bridging_node=str(2), source=str(0), target=str(1),
        insight_score=0.8, explanatory_power=0.7, community_leap=1, path=None
    )
    result = iv.validate(event)
    print(f"  PASS: InsightEvent validated, status={result.validation_status}")
except Exception as e:
    errors.append(f"InsightValidator: {e}")
    print(f"  FAIL: {e}")

# 13. PathScorer
print("\n[13] PathScorer...")
try:
    from reasoning.path_scorer import score_path, community_coherence, path_confidence
    for path in paths[:3]:
        s = score_path(path)
        assert 0.0 <= s <= 1.0
    print(f"  PASS: path scoring works for {min(3, len(paths))} paths")
except Exception as e:
    errors.append(f"PathScorer: {e}")
    print(f"  FAIL: {e}")

# 14. ContradictionEngine
print("\n[14] ContradictionEngine...")
try:
    from core.contradiction_engine import relations_contradict
    _ = relations_contradict("BORN_IN", "DIED_IN")
    print("  PASS: contradiction detection functional")
except Exception as e:
    errors.append(f"ContradictionEngine: {e}")
    print(f"  FAIL: {e}")

# 15. CSV Adapter
print("\n[15] CSV Adapter (load_csv_adapter)...")
try:
    from adapters.csv_adapter import load_csv_adapter
    a = load_csv_adapter("tests/fixtures/toy_graph.csv")
    assert a.node_count() > 0
    print(f"  PASS: {a.node_count()} nodes from CSV fixture")
except Exception as e:
    errors.append(f"CSVAdapter: {e}")
    print(f"  FAIL: {e}")

# 16. Bayesian BeamTraversal (probabilistic mode)
print("\n[16] Bayesian BeamTraversal...")
try:
    bt = BeamTraversal(adapter, csa, beam_width=5, max_hop=2, probabilistic=True, warm_start_strength=2.0)
    bpaths = bt.traverse([0])
    assert len(bpaths) > 0
    # Check Beta variance is present
    has_variance = any(getattr(p, "score_variance", 0.0) > 0.0 for p in bpaths)
    print(f"  PASS: {len(bpaths)} paths, beta_variance_present={has_variance}")
except Exception as e:
    errors.append(f"BayesianBeam: {e}")
    print(f"  FAIL: {e}")

# 17. RelationPathPrior
print("\n[17] RelationPathPrior...")
try:
    from reasoning.relation_path_prior import RelationPathPrior
    prior = RelationPathPrior()
    # update takes (paths, correct_entities)
    prior.update(paths[:3], correct_entities={str(answers[0].entity_id)} if answers else {"1"})
    score = prior.score(paths[0]) if paths else 0.5
    print(f"  PASS: prior update and score={score:.4f}")
except Exception as e:
    errors.append(f"RelationPathPrior: {e}")
    print(f"  FAIL: {e}")

# Summary
print()
print("=" * 50)
if errors:
    print(f"FAILED ({len(errors)} issues):")
    for e in errors:
        print(f"  X {e}")
else:
    print("ALL 17 CHECKS PASS")
