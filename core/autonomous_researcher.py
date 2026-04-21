"""
AutonomousResearcher — Phase 102.
Inspired by Karpathy/Autoresearch.

This daemon autonomously identifies "Magic Constants" in CEREBRUM's source,
generates variants, benchmarks them, and commits the winners.

Supports structural code evolution (logic changes).
"""
import os
import sys
import re
import json
import logging
import subprocess
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s'
)
logger = logging.getLogger("cerebrum.researcher")

class CodeMutation:
    def __init__(self, file_path: str, old_val: str, new_val: str, context: str, is_structural: bool = False):
        self.file_path = file_path
        self.old_val = old_val
        self.new_val = new_val
        self.context = context
        self.is_structural = is_structural

class AutonomousResearcher:
    def __init__(
        self,
        target_files: List[str] = None,
        modulator: Optional[Any] = None,
        recursive_synthesis: bool = True,
        metaplasticity: bool = True,
    ):
        self.target_files = target_files or [
            "core/attention_engine.py",
            "core/chemical_modulator.py",
            "core/reasoning_logit.py",
            "reasoning/traversal.py"
        ]
        self.modulator = modulator
        self.recursive_synthesis = recursive_synthesis
        self.metaplasticity = metaplasticity

        # Evolutionary Hooks for structural changes
        self.hooks = {
            "core/reasoning_logit.py": {
                "hook": "def score",
            }
        }
        self.history_path = Path("research/mutation_history.json")
        self.history_path.parent.mkdir(exist_ok=True)
        self.history = self._load_history()
        
        self.blacklist_path = Path("research/mutation_blacklist.json")
        self.blacklist: Set[str] = set(self._load_json(self.blacklist_path, []))
        
        # Synthetic Library (Phase 105)
        self.synthetic_templates = {
            "recall_bottleneck_detected": {
                "name": "StructuralEntropyPruner",
                "code": """
import numpy as np
from typing import List

class StructuralEntropyPruner:
    \"\"\"Autonomously synthesized module to address recall bottlenecks.\"\"\"
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def prune(self, paths: List[Any]) -> List[Any]:
        if not paths: return paths
        scores = np.array([p.score for p in paths])
        entropy = -np.sum(scores * np.log(scores + 1e-9))
        if entropy > self.threshold:
            # High entropy means search is scattered; prune more aggressively
            return sorted(paths, key=lambda p: p.score, reverse=True)[:max(1, len(paths)//2)]
        return paths
"""
            }
        }

    def _load_history(self) -> List[Dict]:
        return self._load_json(self.history_path, [])

    def _load_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return default
        return default

    def _save_history(self):
        self.history_path.write_text(json.dumps(self.history, indent=2))

    def _save_blacklist(self):
        self.blacklist_path.write_text(json.dumps(list(self.blacklist), indent=2))

    def find_magic_constants(self, file_path: str) -> List[Dict[str, Any]]:
        content = Path(file_path).read_text()
        matches = re.finditer(r"([a-zA-Z_]+)\s*=\s*([0-9\.]+)", content)
        constants = []
        for m in matches:
            name, val = m.groups()
            try:
                fval = float(val)
                if 0.0 < fval < 2.0 or (fval < 100 and fval != int(fval)):
                    constants.append({"name": name, "value": val, "file": file_path})
            except ValueError:
                continue
        return constants

    def propose_structural_change(self, file_path: str) -> Optional[CodeMutation]:
        if not self.recursive_synthesis:
            return None
        if file_path not in self.hooks:
            return None
            
        # Dynamic mutation rate based on Arousal (Phase 104)
        mutation_rate = 0.3
        if self.modulator and self.metaplasticity:
            evo_params = self.modulator.modulate_evolution()
            mutation_rate = evo_params.get("mutation_rate", 0.3)
            
        if random.random() > mutation_rate:
            return None

        content = Path(file_path).read_text()
        if "core/reasoning_logit.py" in file_path:
            pattern = r"raw = \((.*?)\)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                old_logic = match.group(0)
                proposal_body = random.choice([
                    "raw = (a * self.sim * (1.0 + self.cs)) + (b * self.cs)",
                    "raw = (a * self.sim) + (b * self.cs) + (0.05 * np.log1p(self.pr_v + 1e-9))",
                ])
                new_logic = f"{proposal_body} + (g * self.etw) - (d * self.nd) + (e * self.hd) + (z * self.pr_v) + (eta * self.td) + (iota * self.nr_v) - (mu * self.sd) + (theta * self.grounding)"
                return CodeMutation(file_path, old_logic, new_logic, "structural_score", is_structural=True)
        return None

    def run_benchmark(self) -> Tuple[float, float]:
        cmd = [sys.executable, "benchmarks/ikgwq_metaqa.py", "--sample", "30", "--levels", "0", "2"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            h10 = 0.0
            lat = 999.0
            h_match = re.search(r"H@10: ([\d\.]+)", result.stdout)
            if h_match: h10 = float(h_match.group(1))
            l_match = re.search(r"\(([\d\.]+)ms/q\)", result.stdout)
            if l_match: lat = float(l_match.group(1))
            return h10, lat
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return 0.0, 999.0

    def apply_mutation(self, mutation: CodeMutation):
        content = Path(mutation.file_path).read_text()
        if mutation.is_structural:
            new_content = content.replace(mutation.old_val, mutation.new_val, 1)
        else:
            new_content = content.replace(f"{mutation.context} = {mutation.old_val}", 
                                          f"{mutation.context} = {mutation.new_val}", 1)
            if new_content == content:
                new_content = content.replace(mutation.old_val, mutation.new_val, 1)
        Path(mutation.file_path).write_text(new_content)

    def revert_mutation(self, mutation: CodeMutation):
        content = Path(mutation.file_path).read_text()
        if mutation.is_structural:
            new_content = content.replace(mutation.new_val, mutation.old_val, 1)
        else:
            new_content = content.replace(f"{mutation.context} = {mutation.new_val}", 
                                          f"{mutation.context} = {mutation.old_val}", 1)
            if new_content == content:
                new_content = content.replace(mutation.new_val, mutation.old_val, 1)
        Path(mutation.file_path).write_text(new_content)

    def synthesize_module(self, insight: DMNInsight) -> Optional[str]:
        """Phase 105: Create a new architectural module to solve a detected gap."""
        if not self.recursive_synthesis:
            return None
        if insight.description not in self.synthetic_templates:
            return None
            
        template = self.synthetic_templates[insight.description]
        target_path = Path("core") / f"autogen_{template['name'].lower()}.py"
        
        if target_path.exists():
            return str(target_path)
            
        logger.info(f"Synthesizing new module: {template['name']} to address {insight.description}")
        target_path.write_text(template['code'])
        
        # Log success
        self.history.append({
            "timestamp": time.ctime(),
            "type": "synthesis",
            "module": template['name'],
            "file": str(target_path),
            "reason": insight.description
        })
        self._save_history()
        return str(target_path)

    def process_dmn_insights(self, insights: List[DMNInsight]):
        """Analyze insights from the Default Mode Network and trigger research tasks."""
        for ins in insights:
            if ins.type == "heuristic":
                # Trigger a synthesis task
                self.synthesize_module(ins)
            elif ins.type == "frontier":
                # Maybe increase arousal/mutation rate for certain files
                pass

    def scan_and_optimize(self, sample_size: int = 2):
        logger.info("Starting Research Cycle...")
        
        # Meta-scaling (Phase 104)
        commit_mult = 1.0
        if self.modulator and self.metaplasticity:
            evo_params = self.modulator.modulate_evolution()
            commit_mult = evo_params.get("commit_threshold_multiplier", 1.0)
            sample_size = evo_params.get("sample_size", sample_size)

        baseline_h10, baseline_lat = self.run_benchmark()
        logger.info(f"Baseline: H@10={baseline_h10:.4f}, Latency={baseline_lat:.1f}ms")

        # 1. Gather candidates
        all_constants = []
        for f in self.target_files:
            all_constants.extend(self.find_magic_constants(f))
        
        structural_candidates = []
        for f in self.target_files:
            mut = self.propose_structural_change(f)
            if mut: structural_candidates.append(mut)

        # 2. Sample
        if structural_candidates:
            candidates = structural_candidates[:1] + [CodeMutation(c['file'], c['value'], f"{float(c['value'])*1.05:.3f}", c['name']) for c in all_constants[:sample_size-1]]
        else:
            random.shuffle(all_constants)
            candidates = [CodeMutation(c['file'], c['value'], f"{float(c['value'])*1.05:.3f}", c['name']) for c in all_constants[:sample_size]]

        for mutation in candidates:
            mutation_id = f"{mutation.file_path}:{mutation.context}:{mutation.new_val}"
            if mutation_id in self.blacklist: continue
            
            logger.info(f"Testing {mutation.context} in {mutation.file_path}")
            self.apply_mutation(mutation)
            
            # Pre-flight
            try:
                subprocess.run([sys.executable, "-c", f"import {mutation.file_path.replace('.py','').replace('/','.')}"], check=True, capture_output=True)
            except Exception:
                logger.error("Mutation caused error. Reverting.")
                self.revert_mutation(mutation)
                continue

            new_h10, new_lat = self.run_benchmark()
            
            # Success logic with metabolic commit threshold (Phase 104)
            # Higher reinforcement makes it easier to accept slight regressions in latency if accuracy is stable.
            improved_accuracy = new_h10 > baseline_h10
            lat_threshold = baseline_lat * (0.97 * commit_mult) # If mult < 1.0, threshold is tighter. If > 1.0, threshold is looser.
            improved_latency = (new_h10 >= baseline_h10 and new_lat < lat_threshold)

            if improved_accuracy or improved_latency:
                logger.info("SUCCESS: Performance improved!")
                self.history.append({"timestamp": time.ctime(), "file": mutation.file_path, "type": "structural" if mutation.is_structural else "constant", "param": mutation.context, "old": mutation.old_val, "new": mutation.new_val, "h10": new_h10, "lat": new_lat})
                self._save_history()
                baseline_h10, baseline_lat = new_h10, new_lat
            else:
                logger.info("Failed. Reverting.")
                self.blacklist.add(mutation_id)
                self._save_blacklist()
                self.revert_mutation(mutation)

    def run_forever(self, interval: int = 60, cycles: int = -1):
        count = 0
        while cycles < 0 or count < cycles:
            self.scan_and_optimize()
            count += 1
            time.sleep(interval)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=-1)
    args = parser.parse_args()
    researcher = AutonomousResearcher()
    if args.continuous:
        researcher.run_forever(interval=args.interval, cycles=args.cycles)
    else:
        researcher.scan_and_optimize()
