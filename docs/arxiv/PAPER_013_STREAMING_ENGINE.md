### 2.2 Streaming Discretizers
Discretizers transform continuous input signals into discrete symbolic edges. The core discretizers include:

- **Threshold Discretizer**: Continuous float stream $\to$ Edge emitted when value crosses threshold $\theta$.
- **STDP-Discretizer**: Spike/event timestamps $\to$ Directional `CAUSES` edge based on temporal co-occurrence ($\Delta t$).
- **Delta-Discretizer**: Rate-of-change signal $\to$ Edge emitted when $|\Delta x / \Delta t| \geq \theta_{rate}$.
- **Windowed-Frequency-Discretizer**: Event counts per window $\to$ Edge emitted when co-occurrence frequency exceeds $f_{min}$.
- **Pattern-Discretizer**: Symbolic event sequence $\to$ Edge emitted when pattern match probability $\geq p_{match}$.

Each discretizer maintains a small internal sliding buffer and is stateless with respect to the adapter graph. This isolation ensures discretizer failures cannot corrupt the persistent Knowledge Graph.
