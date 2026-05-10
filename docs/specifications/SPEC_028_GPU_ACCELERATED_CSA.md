# SPEC-028: GPU-Accelerated Community-Structured Attention (CSA)

**Status**: v2.51.0 (Phase 167 COMPLETE)
**Author**: Bryan Alexander Buchorn
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Date**: May 5, 2026

---

## 1. Requirement Overview

The core Community-Structured Attention (CSA) formula involves multiple semantic and structural signals. While the vectorized NumPy implementation (SPEC-020) provides a 10x speedup on CPU, extremely large graphs with high average degrees require GPU acceleration to maintain sub-millisecond reasoning latency. This specification defines the requirements for a PyTorch/CUDA-based CSA scoring engine.

## 2. Technical Requirements

### 2.1 Kernel Design
-   **Parallel Scoring**: The scoring function must be implemented as a fused CUDA kernel or a series of optimized PyTorch tensor operations.
-   **Batch Processing**: The engine must support scoring multiple candidate paths simultaneously across a unified tensor.
-   **Sparse Matrix Support**: Neighbor lookups should utilize sparse matrix-vector multiplication (SpMV) where appropriate to minimize VRAM usage.

### 2.2 Formula Components (GPU Implementation)
The scoring engine must compute the 10-parameter formula using GPU primitives:
-   **Cosine Similarity**: Torch-native `cosine_similarity`.
-   **Community Lookups**: Integer-indexed community tensors for $O(1)$ GPU lookup.
-   **Structural Features**: Pre-loaded structural feature tensors (PageRank, Degree).
-   **Metabolic Signals**: Dynamic scalar modulation applied via element-wise multiplication.

## 3. Architectural Design

### 3.1 Data Flow
1.  **Ingestion**: Graph topology and structural features are transferred to VRAM during the `build()` phase.
2.  **Query Path**: Seed entities are mapped to their VRAM indices.
3.  **Expansion**: Candidate neighbors are identified via a sparse adjacency matrix.
4.  **Scoring**: The CSA kernel computes weights for all candidates in a single pass.
5.  **Pruning**: Top-B candidates are selected using GPU-based `topk`.

### 3.2 Memory Management
-   **Hybrid Memory**: Use `Unified Memory` (CUDA managed memory) to allow for graphs larger than VRAM.
-   **FP16/BF16 Support**: Support half-precision embeddings to double the effective graph size on GPU.

## 4. Performance Goals
-   **Latency**: < 5ms for 3-hop queries on graphs with 1M+ edges.
-   **Throughput**: 100+ concurrent queries per second on a single A100 GPU.

## 5. Verification Plan
-   **Unit Tests**: Compare GPU scores against the golden NumPy implementation for parity.
-   **Benchmarking**: Measure latency scaling from 100 to 1,000,000 nodes.
-   **Memory Audit**: Ensure zero VRAM leaks during continuous query loops.

---
**Specification Finalized: v2.51.0 (Phase 167 COMPLETE)**
