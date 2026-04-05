
from core.parameter_learner import MetaParameterLearner
from reasoning.traversal import TraversalPath

def test_meta_parameter_adaptation():
    """
    Verify that MetaParameterLearner adapts community parameters
    based on positive and negative feedback (10-parameter, Phase 45).
    """
    # 1. Setup learner with default prior (10 params)
    prior = (0.5,) * 10
    learner = MetaParameterLearner(global_prior=prior, learning_rate=0.1)

    # Community 1 is our target
    cid = 1

    # 2. Simulate a path through community 1
    # Features: (sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding)
    path = TraversalPath(
        nodes=["A", "rel", "B"],
        edge_features=[(0.9, 1.0, 0.1, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0)],
        community_sequence=[cid]
    )

    # 3. Apply POSITIVE feedback — alpha (sim) and beta (cs) should increase
    learner.update_from_feedback(path, reward=1.0)

    params_after_pos = learner.get_params(cid)
    assert len(params_after_pos) == 10
    assert params_after_pos[0] > prior[0], "alpha should increase after positive feedback"
    assert params_after_pos[1] > prior[1], "beta should increase after positive feedback"

    # 4. Apply NEGATIVE feedback — parameters should decrease back
    learner.update_from_feedback(path, reward=-1.0)
    learner.update_from_feedback(path, reward=-1.0)  # double dose

    params_after_neg = learner.get_params(cid)
    assert params_after_neg[0] < params_after_pos[0], "alpha should decrease after negative feedback"

    print("Meta-learning adaptation verified (10-parameter).")


def test_meta_parameter_legacy_5element_compat():
    """
    Verify that MetaParameterLearner gracefully handles legacy 5-element
    edge_features (zero-padded internally).
    """
    prior = (0.5,) * 10
    learner = MetaParameterLearner(global_prior=prior, learning_rate=0.1)
    cid = 2

    path = TraversalPath(
        nodes=["X", "rel", "Y"],
        edge_features=[(0.9, 1.0, 0.1, 0.0, 0.5)],  # legacy 5-element
        community_sequence=[cid]
    )

    # Should not raise
    learner.update_from_feedback(path, reward=1.0)
    params = learner.get_params(cid)
    assert len(params) == 10


if __name__ == "__main__":
    test_meta_parameter_adaptation()
    test_meta_parameter_legacy_5element_compat()
