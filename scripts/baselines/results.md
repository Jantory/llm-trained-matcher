# Baseline Results

Results path: `results/baselines/openai/<dataset>/<model>/training_log.json`

---

## MatchGPT — gpt-4.1-mini

Zero-shot generative matching. Prompt: *"Determine whether the following two entities refer to the same real-world entity. Answer yes or no."*

| Dataset | F1     | Precision | Recall |
|---------|--------|-----------|--------|
| ab      | 0.8718 | 0.8386    | 0.9078 |
| ag      | 0.6464 | 0.5350    | 0.8162 |
| ds      | 0.8736 | 0.8991    | 0.8495 |
| rt      | 0.6954 | 0.7215    | 0.6712 |
| sh      | 0.9400 | 1.0000    | 0.8868 |
| sr      | 0.8720 | 0.9862    | 0.7814 |
| wa      | 0.7824 | 0.8353    | 0.7358 |
| w20     | 0.8730 | 0.9183    | 0.8320 |
| w80     | 0.8215 | 0.8333    | 0.8100 |
