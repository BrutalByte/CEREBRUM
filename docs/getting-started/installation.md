# Installation

## Requirements

- Python 3.10+
- 4GB+ RAM (16GB+ recommended for large KBs)
- GPU optional but recommended (NVIDIA, AMD, Apple Silicon, Intel Gaudi all supported)

## pip Install

```bash
# Minimal (no embeddings, no API server)
pip install cerebrum-kg

# With semantic embeddings (BGE)
pip install "cerebrum-kg[embeddings]"

# With REST API server
pip install "cerebrum-kg[api]"

# With Neo4j adapter
pip install "cerebrum-kg[neo4j]"

# Everything
pip install "cerebrum-kg[all]"
```

## GPU

```bash
# NVIDIA CUDA 12.x
pip install "cerebrum-kg[all]" torch --index-url https://download.pytorch.org/whl/cu124

# NVIDIA CUDA 11.8
pip install "cerebrum-kg[all]" torch --index-url https://download.pytorch.org/whl/cu118

# AMD ROCm 6.0
pip install "cerebrum-kg[all]" torch --index-url https://download.pytorch.org/whl/rocm6.0

# CPU-only
pip install "cerebrum-kg[all]"
```

## Docker

```bash
docker pull ghcr.io/brutalbyte/cerebrum:latest

docker run -p 8200:8200 \
  -v $(pwd)/data:/data \
  -e CEREBRUM_CSV_PATH=/data/kb.csv \
  ghcr.io/brutalbyte/cerebrum:latest
```

## From Source

```bash
git clone https://github.com/BrutalByte/CEREBRUM.git
cd CEREBRUM
pip install -e ".[all]"
```

## Verify

```bash
cerebrum init --demo
```

Expected output:
```
KB Ready — 21 entities | 30 relations | 3 communities
```
