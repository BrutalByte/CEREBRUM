import os

def edit_cerebrum():
    path = '../core/cerebrum.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add looped definition
    old_code = """        # Phase 99: Thalamic Gating - build WM priming map
        _priming_map = self._build_priming_map()

        try:
            if max_loops > 1:
                paths = looped.traverse(seeds, query_embedding=query_embedding, trace_info=trace_info, node_priming=_priming_map if _priming_map else None)[0]
            else:
                paths = traversal.traverse(seeds, query_embedding=query_embedding, trace_info=trace_info, node_priming=_priming_map if _priming_map else None)"""
    
    new_code = """        # Phase 99: Thalamic Gating - build WM priming map
        _priming_map = self._build_priming_map()

        # Phase 70: Looped Reasoning (Recursive refinement)
        looped = None
        if max_loops > 1:
            from reasoning.looped_traversal import LoopedBeamTraversal
            looped = LoopedBeamTraversal(
                traversal=traversal,
                predictive_coder=self.predictive_coder,
                max_loops=max_loops
            )

        try:
            if max_loops > 1 and looped is not None:
                paths = looped.traverse(seeds, query_embedding=query_embedding, trace_info=trace_info, node_priming=_priming_map if _priming_map else None)[0]
            else:
                paths = traversal.traverse(seeds, query_embedding=query_embedding, trace_info=trace_info, node_priming=_priming_map if _priming_map else None)"""
    
    content = content.replace(old_code, new_code)
    
    # 2. Fix finally block redundancy
    old_finally = """        finally:
            # Phase 68: Natural decay of hormonal state
            self.modulator.step()
            if not needs_custom:
                self._traversal._beam_widths = _prev_widths
            # Phase 68: Natural decay of hormonal state after query completion
            self.modulator.step()
            # Phase 136: restore shared traversal's beam widths
            if not needs_custom:
                self._traversal._beam_widths = _prev_widths"""
    
    new_finally = """        finally:
            # Phase 68: Natural decay of hormonal state
            self.modulator.step()
            # Phase 136: restore shared traversal's beam widths
            if not needs_custom:
                self._traversal._beam_widths = _prev_widths"""
    
    # 3. Fix set_research_agent to set _research_agent
    old_set_agent = """    def set_research_agent(self, agent: Any) -> None:
        self.research_agent = agent
        if hasattr(agent, "_adapter"):
            agent._adapter.graph = self"""
    
    new_set_agent = """    def set_research_agent(self, agent: Any) -> None:
        self.research_agent = agent
        self._research_agent = agent
        if hasattr(agent, "_adapter"):
            agent._adapter.graph = self"""
    
    content = content.replace(old_set_agent, new_set_agent)
    
    # 4. Add debug prints
    debug_code = """
            # Phase 150: Frontal Engine executive strategy
            entropy = getattr(traversal, "_last_entropy", 0.0)
            strategy = self.frontal.determine_strategy(paths, entropy=entropy)
            print(f"DEBUG: strategy={strategy}, agent={self._research_agent}, gaps={getattr(traversal, 'epistemic_gaps', [])}")
"""
    content = content.replace("""
            # Phase 150: Frontal Engine executive strategy
            entropy = getattr(traversal, "_last_entropy", 0.0)
            strategy = self.frontal.determine_strategy(paths, entropy=entropy)""", debug_code)
    
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    print("Successfully edited core/cerebrum.py with debug prints")

if __name__ == "__main__":
    edit_cerebrum()
