Loading Baseline features from experiments/tuev_lp/features/recon_features.npz...
Loading Flagship features from experiments/tuev_lp/features/neuro_ke_features.npz...
Loading FeatOnly features from experiments/tuev_lp_feat_only/features/feat_only_features.npz...
Total Unique Training Subjects: 232

=== Comparative Results Table ===
| Ratio    | NumSub | Model        | Feature    | Dim   | Acc     | BAcc    |
|----------|--------|--------------|------------|-------|---------|---------|
| 0.5%     | 1      | SKIPPED (Only 1 class) |
| 1.0%     | 2      | Baseline     | EEG        | 200   | 42.96% | 21.21% |
| 1.0%     | 2      | Neuro-KE     | eeg        | 200   | 40.13% | 25.00% |
| 1.0%     | 2      | Neuro-KE     | feat       | 200   | 43.55% | 24.47% |
| 1.0%     | 2      | Neuro-KE     | full       | 400   | 39.13% | 24.12% |
| 1.0%     | 2      | Neuro-KE     | pred       | 62    | 38.95% | 23.20% |
| 1.0%     | 2      | FeatOnly     | eeg        | 200   | 42.72% | 24.64% |
| 1.0%     | 2      | FeatOnly     | feat       | 200   | 42.07% | 23.02% |
| 1.0%     | 2      | FeatOnly     | full       | 400   | 43.49% | 23.52% |
| 1.0%     | 2      | FeatOnly     | pred       | 62    | 42.25% | 23.88% |
| 5.0%     | 11     | Baseline     | EEG        | 200   | 40.42% | 25.01% |
| 5.0%     | 11     | Neuro-KE     | eeg        | 200   | 47.85% | 29.35% |
| 5.0%     | 11     | Neuro-KE     | feat       | 200   | 47.91% | 30.63% |
| 5.0%     | 11     | Neuro-KE     | full       | 400   | 48.85% | 30.70% |
| 5.0%     | 11     | Neuro-KE     | pred       | 62    | 46.55% | 29.04% |
| 5.0%     | 11     | FeatOnly     | eeg        | 200   | 50.85% | 33.50% |
| 5.0%     | 11     | FeatOnly     | feat       | 200   | 50.27% | 32.44% |
| 5.0%     | 11     | FeatOnly     | full       | 400   | 51.38% | 34.73% |
| 5.0%     | 11     | FeatOnly     | pred       | 62    | 50.68% | 31.32% |
| 10.0%    | 23     | Baseline     | EEG        | 200   | 38.77% | 24.84% |
| 10.0%    | 23     | Neuro-KE     | eeg        | 200   | 47.08% | 32.42% |
| 10.0%    | 23     | Neuro-KE     | feat       | 200   | 46.91% | 31.84% |
| 10.0%    | 23     | Neuro-KE     | full       | 400   | 47.85% | 32.94% |
| 10.0%    | 23     | Neuro-KE     | pred       | 62    | 47.67% | 29.97% |
| 10.0%    | 23     | FeatOnly     | eeg        | 200   | 53.39% | 34.74% |
| 10.0%    | 23     | FeatOnly     | feat       | 200   | 53.80% | 38.36% |
| 10.0%    | 23     | FeatOnly     | full       | 400   | 52.56% | 34.22% |
| 10.0%    | 23     | FeatOnly     | pred       | 62    | 53.45% | 37.89% |
| 20.0%    | 46     | Baseline     | EEG        | 200   | 47.14% | 31.47% |
| 20.0%    | 46     | Neuro-KE     | eeg        | 200   | 55.04% | 38.57% |
| 20.0%    | 46     | Neuro-KE     | feat       | 200   | 56.22% | 40.55% |
| 20.0%    | 46     | Neuro-KE     | full       | 400   | 53.15% | 37.19% |
| 20.0%    | 46     | Neuro-KE     | pred       | 62    | 53.27% | 34.78% |
| 20.0%    | 46     | FeatOnly     | eeg        | 200   | 56.75% | 39.19% |
| 20.0%    | 46     | FeatOnly     | feat       | 200   | 59.75% | 41.18% |
| 20.0%    | 46     | FeatOnly     | full       | 400   | 61.58% | 43.35% |
| 20.0%    | 46     | FeatOnly     | pred       | 62    | 63.17% | 43.22% |
| 50.0%    | 116    | Baseline     | EEG        | 200   | 51.86% | 32.78% |
| 50.0%    | 116    | Neuro-KE     | eeg        | 200   | 57.40% | 39.79% |
| 50.0%    | 116    | Neuro-KE     | feat       | 200   | 58.69% | 40.61% |
| 50.0%    | 116    | Neuro-KE     | full       | 400   | 58.99% | 41.51% |
| 50.0%    | 116    | Neuro-KE     | pred       | 62    | 54.86% | 37.54% |
| 50.0%    | 116    | FeatOnly     | eeg        | 200   | 66.29% | 45.79% |
| 50.0%    | 116    | FeatOnly     | feat       | 200   | 67.12% | 46.97% |
| 50.0%    | 116    | FeatOnly     | full       | 400   | 63.11% | 47.47% |
| 50.0%    | 116    | FeatOnly     | pred       | 62    | 67.12% | 47.12% |
| 100%     | 232    | Baseline     | EEG        | 200   | 47.79% | 32.58% |
| 100%     | 232    | Neuro-KE     | eeg        | 200   | 52.86% | 39.67% |
| 100%     | 232    | Neuro-KE     | feat       | 200   | 53.92% | 41.19% |
| 100%     | 232    | Neuro-KE     | full       | 400   | 55.39% | 42.42% |
| 100%     | 232    | Neuro-KE     | pred       | 62    | 50.15% | 37.34% |
| 100%     | 232    | FeatOnly     | eeg        | 200   | 65.23% | 47.79% |
| 100%     | 232    | FeatOnly     | feat       | 200   | 63.76% | 47.68% |
| 100%     | 232    | FeatOnly     | full       | 400   | 64.23% | 48.89% |
| 100%     | 232    | FeatOnly     | pred       | 62    | 59.75% | 48.17% |
