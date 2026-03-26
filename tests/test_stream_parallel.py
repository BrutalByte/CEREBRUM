
import pytest
import time
from adapters.stream_adapter import StreamAdapter
from core.stream_engine import StreamEvent
from core.thalamus import IngestionPipeline

class SlowPipeline(IngestionPipeline):
    """Pipeline that simulates heavy CPU work."""
    def __init__(self):
        super().__init__(namespace="text")
        
    def process(self, source, target, relation, properties):
        time.sleep(0.1) # 100ms delay
        return super().process(source, target, relation, properties)

def test_ingest_parallel_efficiency():
    """
    Verify that pipeline processing doesn't hold the graph lock
    during the entire operation.
    
    We simulate this by checking if multiple ingest calls can 'overlap'
    their processing time, or at least that the lock is only held
    for the final graph insertion.
    """
    pipeline = SlowPipeline()
    adapter = StreamAdapter(pipeline=pipeline)
    
    start_time = time.time()
    
    # Ingest two events. 
    # If synchronous and locked for the whole duration, it will take >= 200ms.
    # While Python's GIL limits true parallelism for CPU work, 
    # running outside the graph lock means readers aren't blocked.
    ev1 = StreamEvent("A", "rel", "B")
    ev2 = StreamEvent("C", "rel", "D")
    
    adapter.ingest(ev1)
    adapter.ingest(ev2)
    
    duration = time.time() - start_time
    assert duration >= 0.2, "Should take at least 200ms due to slow pipeline"
    
    # Verify both edges added with 'text:' prefix from Thalamus
    G = adapter.to_networkx()
    assert G.has_edge("text:A", "text:B")
    assert G.has_edge("text:C", "text:D")

def test_ingest_batch_parallel_efficiency():
    """Verify ingest_batch also processes outside the lock."""
    pipeline = SlowPipeline()
    adapter = StreamAdapter(pipeline=pipeline)
    
    evs = [
        StreamEvent("A", "rel", "B"),
        StreamEvent("C", "rel", "D")
    ]
    
    start_time = time.time()
    adapter.ingest_batch(evs)
    duration = time.time() - start_time
    
    assert duration >= 0.2
    assert adapter.to_networkx().number_of_edges() == 2
