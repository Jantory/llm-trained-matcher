# LR Sweep Results — un dataset (8B models)

WandB project: `em-matcher-lr-sweep-8b`  
Results path: `results/lr_sweep_8b/<paradigm>/un/<model>_lr<lr>/training_log.json`  
Collect: `bash scripts/lr_sweep_8b/collect_results.sh`

---

## Bi-Encoder (LRs: 1e-5, 3e-5, 5e-5, 1e-4, 2e-4)

| Model | LR | F1 | P | R |
|---|---|---|---|---|
| Qwen3-8B | 1e-5 | 0.2000 | 0.2143 | 0.1875 |
| Qwen3-8B | 3e-5 | **0.4556** | 0.3802 | 0.5682 |
| Qwen3-8B | 5e-5 | 0.1830 | 0.2154 | 0.1591 |
| Qwen3-8B | 1e-4 | 0.3691 | 0.3289 | 0.4205 |
| Qwen3-8B | 2e-4 | 0.3633 | 0.2871 | 0.4943 |
| Qwen3-8B-Base | 1e-5 | 0.3386 | 0.2607 | 0.4830 |
| Qwen3-8B-Base | 3e-5 | 0.4205 | 0.3691 | 0.4886 |
| Qwen3-8B-Base | 5e-5 | 0.4332 | 0.4534 | 0.4148 |
| Qwen3-8B-Base | 1e-4 | 0.3720 | 0.3025 | 0.4830 |
| Qwen3-8B-Base | 2e-4 | **0.5753** | 0.5556 | 0.5966 |
| Qwen3-Embedding-8B | 1e-5 | 0.8213 | 0.7739 | 0.8750 |
| Qwen3-Embedding-8B | 3e-5 | 0.8249 | 0.8202 | 0.8295 |
| Qwen3-Embedding-8B | 5e-5 | 0.7965 | 0.8155 | 0.7784 |
| Qwen3-Embedding-8B | 1e-4 | **0.8272** | 0.8249 | 0.8295 |
| Qwen3-Embedding-8B | 2e-4 | 0.8177 | 0.7957 | 0.8409 |

## Cross-Encoder (LRs: 1e-5, 3e-5, 5e-5, 1e-4, 2e-4)

| Model | LR | F1 | P | R |
|---|---|---|---|---|
| Qwen3-8B | 1e-5 | 0.7273 | 0.6869 | 0.7727 |
| Qwen3-8B | 3e-5 | 0.8639 | 0.9012 | 0.8295 |
| Qwen3-8B | 5e-5 | 0.8629 | 0.8678 | 0.8580 |
| Qwen3-8B | 1e-4 | 0.8260 | 0.8589 | 0.7955 |
| Qwen3-8B | 2e-4 | **0.8941** | 0.9268 | 0.8636 |
| Qwen3-8B-Base | 1e-5 | 0.5031 | 0.5548 | 0.4602 |
| Qwen3-8B-Base | 3e-5 | 0.7895 | 0.8133 | 0.7670 |
| Qwen3-8B-Base | 5e-5 | 0.8406 | 0.8580 | 0.8239 |
| Qwen3-8B-Base | 1e-4 | 0.8208 | 0.8353 | 0.8068 |
| Qwen3-8B-Base | 2e-4 | **0.8778** | 0.8587 | 0.8977 |
| Qwen3-Embedding-8B | 1e-5 | 0.7956 | 0.7742 | 0.8182 |
| Qwen3-Embedding-8B | 3e-5 | 0.8693 | 0.8191 | 0.9261 |
| Qwen3-Embedding-8B | 5e-5 | **0.8901** | 0.8827 | 0.8977 |
| Qwen3-Embedding-8B | 1e-4 | 0.8780 | 0.8394 | 0.9205 |
| Qwen3-Embedding-8B | 2e-4 | 0.8800 | 0.8851 | 0.8750 |

## Generative (LRs: 3e-5, 5e-5, 1e-4, 2e-4, 3e-4)

| Model | LR | F1 | P | R |
|---|---|---|---|---|
| Qwen3-8B | 3e-5 | 0.8595 | 0.8342 | 0.8864 |
| Qwen3-8B | 5e-5 | **0.8975** | 0.8757 | 0.9205 |
| Qwen3-8B | 1e-4 | 0.8833 | 0.8641 | 0.9034 |
| Qwen3-8B | 2e-4 | 0.8908 | 0.9012 | 0.8807 |
| Qwen3-8B | 3e-4 | 0.8966 | 0.9070 | 0.8864 |
| Qwen3-8B-Base | 3e-5 | 0.8680 | 0.8970 | 0.8409 |
| Qwen3-8B-Base | 5e-5 | 0.8844 | 0.9000 | 0.8693 |
| Qwen3-8B-Base | 1e-4 | 0.8933 | 0.8833 | 0.9034 |
| Qwen3-8B-Base | 2e-4 | 0.8798 | 0.8474 | 0.9148 |
| Qwen3-8B-Base | 3e-4 | **0.9174** | 0.9200 | 0.9148 |
| Qwen3-Embedding-8B | 3e-5 | 0.8617 | 0.8100 | 0.9205 |
| Qwen3-Embedding-8B | 5e-5 | **0.9096** | 0.9341 | 0.8864 |
| Qwen3-Embedding-8B | 1e-4 | 0.8889 | 0.8914 | 0.8864 |
| Qwen3-Embedding-8B | 2e-4 | 0.9023 | 0.9128 | 0.8920 |
| Qwen3-Embedding-8B | 3e-4 | 0.8977 | 0.8977 | 0.8977 |
