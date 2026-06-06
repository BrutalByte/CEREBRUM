---
name: tuner-launch
description: Launch an Optuna hyperparameter tuning run for MetaQA. Configures trial budget, sample size, and MLflow logging. Usage: /tuner-launch [--trials N] [--sample N]
disable-model-invocation: true
---

Launch a CEREBRUM Optuna tuning run:

```bash
cd E:/Development/Cerebrum

# Parse optional args (defaults: 30 trials, 500 questions per trial)
TRIALS=$(echo "${ARGS}" | grep -oP '(?<=--trials )\d+' || echo "30")
SAMPLE=$(echo "${ARGS}" | grep -oP '(?<=--sample )\d+' || echo "500")

echo "Launching tuner: ${TRIALS} trials × ${SAMPLE} questions"
echo "Output: benchmarks/tuner_$(date +%Y%m%dT%H%M%S).jsonl"
echo "MLflow UI: mlflow ui --port 5000"

python benchmarks/cerebrum_tuner.py \
  --trials "${TRIALS}" \
  --sample "${SAMPLE}" \
  --mlflow
```

Canonical Phase 185/186 baseline to beat: H@1=56.12%, MRR=0.6704

After the run completes, use `/benchmark-report` to analyze the results.

Notes:
- Each trial uses 8 workers; estimated time: ~70s per trial on RTX 5090
- Results saved to `benchmarks/tuner_TIMESTAMP.jsonl`
- Optuna study persisted to `optuna_studies/` (excluded from git)
