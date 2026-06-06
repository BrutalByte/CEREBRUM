---
name: orin-deploy
description: Sync CEREBRUM to AGX Orin satellite node (agxorin.local) via SSH and restart the server.
disable-model-invocation: true
---

Deploy to AGX Orin:

```bash
# Sync repo (excluding large data files and benchmarks)
rsync -avz --progress \
  --exclude='.git' \
  --exclude='data/' \
  --exclude='benchmarks/tuner_*.jsonl' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='mlruns/' \
  E:/Development/Cerebrum/ \
  brutalbyte@agxorin.local:/home/brutalbyte/cerebrum/

# Restart server on Orin
ssh brutalbyte@agxorin.local "cd /home/brutalbyte/cerebrum && bash startup_cerebrum.sh"
```

Orin connection: `brutalbyte@agxorin.local` (password: CEREBRUM)
See reference: `memory/reference_orin_ssh.md`
