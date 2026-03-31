
from core.parameter_learner import MetaParameterLearner
from reasoning.traversal import TraversalPath

def test_meta_parameter_adaptation():
    """
    Verify that MetaParameterLearner adapts community parameters 
    based on positive and negative feedback.
    """
    # 1. Setup learner with default prior
    prior = (0.5, 0.5, 0.5, 0.5, 0.5)
    learner = MetaParameterLearner(global_prior=prior, learning_rate=0.1)
    
    # Community 1 is our target
    cid = 1
    
    # 2. Simulate a path through community 1
    # Features: (sim, cs, etw, nd, hd)
    # Let's say high sim was present
    path = TraversalPath(
        nodes=["A", "rel", "B"],
        edge_features=[(0.9, 1.0, 0.1, 0.0, 0.5)],
        community_sequence=[cid]
    )
    
    # 3. Apply POSITIVE feedback
    # Should INCREASE parameters that correlate with the features
    learner.update_from_feedback(path, reward=1.0)
    
    params_after_pos = learner.get_params(cid)
    print(f"Params after positive: {params_after_pos}")
    
    # alpha (sim) and beta (cs) should have increased
    assert params_after_pos[0] > prior[0]
    assert params_after_pos[1] > prior[1]
    
    # 4. Apply NEGATIVE feedback
    # Should DECREASE parameters
    learner.update_from_feedback(path, reward=-1.0)
    learner.update_from_feedback(path, reward=-1.0) # Double dose
    
    params_after_neg = learner.get_params(cid)
    print(f"Params after negative: {params_after_neg}")
    
    assert params_after_neg[0] < params_after_pos[0]
    
    print("Meta-learning adaptation verified.")

if __name__ == "__main__":
    test_meta_parameter_adaptation()
