import os

def patch_file(path, target_pattern, replacement):
    with open(path, 'r') as f:
        content = f.read()
    if target_pattern in content:
        new_content = content.replace(target_pattern, replacement)
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"Patched {path}")
    else:
        print(f"Pattern not found in {path}")

# Patch RemoteCerebrumAdapter
remote_target = """    def find_similar(
        self, 
        embedding: "np.ndarray", 
        top_k: int = 10
    ) -> List[Entity]:"""
remote_repl = """    def add_edge(
        self,
        u: str,
        v: str,
        relation: str,
        confidence: float = 1.0,
        provenance: str = "",
        synthetic: bool = False,
    ) -> None:
        import logging
        logging.getLogger("cerebrum.remote_adapter").warning("RemoteCerebrumAdapter: Materialization not supported.")

    def find_similar(
        self, 
        embedding: "np.ndarray", 
        top_k: int = 10
    ) -> List[Entity]:"""

patch_file('E:\\Development\\Cerebrum\\adapters\\remote_adapter.py', remote_target, remote_repl)

# Patch FederatedAdapter (add_edge)
# Locate: "    def get_embedding"
fed_target = """    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:"""
fed_repl = """    def add_edge(
        self,
        u: str,
        v: str,
        relation: str,
        confidence: float = 1.0,
        provenance: str = "",
        synthetic: bool = False,
    ) -> None:
        # Fallback to first available adapter
        if self.adapters:
            first_name = list(self.adapters.keys())[0]
            self.adapters[first_name].add_edge(u, v, relation, confidence, provenance, synthetic)

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:"""

patch_file('E:\\Development\\Cerebrum\\adapters\\federated_adapter.py', fed_target, fed_repl)
