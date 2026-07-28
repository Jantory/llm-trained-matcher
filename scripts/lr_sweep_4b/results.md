# LR Sweep Results — un dataset (4B models)

WandB project: `em-matcher-lr-sweep-4b`  
Results path: `results/lr_sweep_4b/<paradigm>/un/<model>_lr<lr>/training_log.json`  
Collect: `bash scripts/lr_sweep_4b/collect_results.sh`

---

## Bi-Encoder (LRs: 1e-5, 3e-5, 5e-5, 1e-4, 2e-4)

| Model | LR | F1 | P | R |
|---|---|---|---|---|
| Qwen3-4B | 1e-5 | 0.2602 | 0.2487 | 0.2727 |
| Qwen3-4B | 3e-5 | 0.3799 | 0.3736 | 0.3864 |
| Qwen3-4B | 5e-5 | 0.6158 | 0.5576 | 0.6875 |
| Qwen3-4B | 1e-4 | 0.6625 | 0.7361 | 0.6023 |
| Qwen3-4B | 2e-4 | **0.6789** | 0.7351 | 0.6307 |
| Qwen3-4B-Base | 1e-5 | 0.4140 | 0.3504 | 0.5057 |
| Qwen3-4B-Base | 3e-5 | 0.4216 | 0.3707 | 0.4886 |
| Qwen3-4B-Base | 5e-5 | 0.3866 | 0.3812 | 0.3920 |
| Qwen3-4B-Base | 1e-4 | 0.5299 | 0.4880 | 0.5795 |
| Qwen3-4B-Base | 2e-4 | **0.6789** | 0.7351 | 0.6307 |
| Qwen3-Embedding-4B | 1e-5 | 0.7968 | 0.7438 | 0.8580 |
| Qwen3-Embedding-4B | 3e-5 | **0.8383** | 0.8861 | 0.7955 |
| Qwen3-Embedding-4B | 5e-5 | 0.8011 | 0.8011 | 0.8011 |
| Qwen3-Embedding-4B | 1e-4 | 0.7956 | 0.7742 | 0.8182 |
| Qwen3-Embedding-4B | 2e-4 | 0.8057 | 0.8103 | 0.8011 |

## Cross-Encoder (LRs: 1e-5, 3e-5, 5e-5, 1e-4, 2e-4)

| Model | LR | F1 | P | R |
|---|---|---|---|---|
| Qwen3-4B | 1e-5 | 0.5804 | 0.6525 | 0.5227 |
| Qwen3-4B | 3e-5 | 0.8256 | 0.8452 | 0.8068 |
| Qwen3-4B | 5e-5 | 0.8698 | 0.8486 | 0.8920 |
| Qwen3-4B | 1e-4 | 0.8750 | 0.8750 | 0.8750 |
| Qwen3-4B | 2e-4 | **0.8789** | 0.8715 | 0.8864 |
| Qwen3-4B-Base | 1e-5 | 0.7128 | 0.6700 | 0.7614 |
| Qwen3-4B-Base | 3e-5 | 0.8304 | 0.8554 | 0.8068 |
| Qwen3-4B-Base | 5e-5 | 0.8650 | 0.8396 | 0.8920 |
| Qwen3-4B-Base | 1e-4 | 0.8708 | 0.8611 | 0.8807 |
| Qwen3-4B-Base | 2e-4 | **0.8981** | 0.8717 | 0.9261 |
| Qwen3-Embedding-4B | 1e-5 | 0.7390 | 0.7636 | 0.7159 |
| Qwen3-Embedding-4B | 3e-5 | 0.8113 | 0.8045 | 0.8182 |
| Qwen3-Embedding-4B | 5e-5 | 0.8563 | 0.8848 | 0.8295 |
| Qwen3-Embedding-4B | 1e-4 | 0.8669 | 0.8644 | 0.8693 |
| Qwen3-Embedding-4B | 2e-4 | **0.8920** | 0.8703 | 0.9148 |

## Generative (LRs: 3e-5, 5e-5, 1e-4, 2e-4, 3e-4)

| Model | LR | F1 | P | R |
|---|---|---|---|---|
| Qwen3-4B | 3e-5 | 0.8366 | 0.8162 | 0.8580 |
| Qwen3-4B | 5e-5 | 0.8728 | 0.8882 | 0.8580 |
| Qwen3-4B | 1e-4 | 0.8669 | 0.8644 | 0.8693 |
| Qwen3-4B | 2e-4 | 0.8671 | 0.8824 | 0.8523 |
| Qwen3-4B | 3e-4 | **0.9036** | 0.8770 | 0.9318 |
| Qwen3-4B-Base | 3e-5 | 0.8604 | 0.8629 | 0.8580 |
| Qwen3-4B-Base | 5e-5 | 0.8736 | 0.8837 | 0.8636 |
| Qwen3-4B-Base | 1e-4 | **0.8870** | 0.8820 | 0.8920 |
| Qwen3-4B-Base | 2e-4 | 0.8791 | 0.8511 | 0.9091 |
| Qwen3-4B-Base | 3e-4 | 0.8844 | 0.9000 | 0.8693 |
| Qwen3-Embedding-4B | 3e-5 | 0.8958 | 0.8883 | 0.9034 |
| Qwen3-Embedding-4B | 5e-5 | **0.8986** | 0.8677 | 0.9318 |
| Qwen3-Embedding-4B | 1e-4 | 0.8956 | 0.8670 | 0.9261 |
| Qwen3-Embedding-4B | 2e-4 | 0.8870 | 0.8820 | 0.8920 |
| Qwen3-Embedding-4B | 3e-4 | 0.8754 | 0.8935 | 0.8580 |
