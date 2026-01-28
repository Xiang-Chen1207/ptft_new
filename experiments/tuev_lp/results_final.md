Loading Baseline features from experiments/tuev_lp/features/recon_features.npz...
Loading Flagship features from experiments/tuev_lp/features/neuro_ke_features.npz...
Loading FeatOnly features from experiments/tuev_lp_feat_only/features/feat_only_features.npz...
Total Unique Training Subjects: 232

=== Comparative Results Table ===
| Ratio    | NumSub | Model        | Feature    | Dim   | Acc     | BAcc    |
|----------|--------|--------------|------------|-------|---------|---------|
| 0.5%     | 1      | SKIPPED (Only 1 class) |
| 1.0%     | 2      | Baseline     | EEG        | 200   | 44.96% | 20.73% |
| 1.0%     | 2      | Neuro-KE     | eeg        | 200   | 37.42% | 27.07% |
| 1.0%     | 2      | Neuro-KE     | feat       | 200   | 45.20% | 23.04% |
| 1.0%     | 2      | Neuro-KE     | full       | 400   | 43.08% | 25.05% |
| 1.0%     | 2      | Neuro-KE     | pred       | 62    | 35.36% | 23.89% |
| 1.0%     | 2      | FeatOnly     | eeg        | 200   | 45.49% | 29.48% |
| 1.0%     | 2      | FeatOnly     | feat       | 200   | 49.91% | 28.25% |
| 1.0%     | 2      | FeatOnly     | full       | 400   | 46.55% | 26.71% |
| 1.0%     | 2      | FeatOnly     | pred       | 62    | 43.90% | 25.31% |
| 5.0%     | 11     | Baseline     | EEG        | 200   | 43.96% | 25.24% |
| 5.0%     | 11     | Neuro-KE     | eeg        | 200   | 50.38% | 32.05% |
| 5.0%     | 11     | Neuro-KE     | feat       | 200   | 53.74% | 35.97% |
| 5.0%     | 11     | Neuro-KE     | full       | 400   | 52.15% | 34.72% |
| 5.0%     | 11     | Neuro-KE     | pred       | 62    | 48.50% | 30.44% |
| 5.0%     | 11     | FeatOnly     | eeg        | 200   | 51.86% | 34.41% |
| 5.0%     | 11     | FeatOnly     | feat       | 200   | 52.21% | 34.22% |
| 5.0%     | 11     | FeatOnly     | full       | 400   | 52.86% | 35.72% |
| 5.0%     | 11     | FeatOnly     | pred       | 62    | 51.38% | 31.76% |
| 10.0%    | 23     | Baseline     | EEG        | 200   | 39.13% | 25.71% |
| 10.0%    | 23     | Neuro-KE     | eeg        | 200   | 54.27% | 37.78% |
| 10.0%    | 23     | Neuro-KE     | feat       | 200   | 51.38% | 31.69% |
| 10.0%    | 23     | Neuro-KE     | full       | 400   | 52.09% | 34.24% |
| 10.0%    | 23     | Neuro-KE     | pred       | 62    | 51.50% | 34.96% |
| 10.0%    | 23     | FeatOnly     | eeg        | 200   | 54.68% | 36.62% |
| 10.0%    | 23     | FeatOnly     | feat       | 200   | 54.15% | 36.87% |
| 10.0%    | 23     | FeatOnly     | full       | 400   | 54.92% | 37.81% |
| 10.0%    | 23     | FeatOnly     | pred       | 62    | 53.09% | 38.08% |
| 20.0%    | 46     | Baseline     | EEG        | 200   | 47.79% | 31.16% |
| 20.0%    | 46     | Neuro-KE     | eeg        | 200   | 63.82% | 46.85% |
| 20.0%    | 46     | Neuro-KE     | feat       | 200   | 60.87% | 41.69% |
| 20.0%    | 46     | Neuro-KE     | full       | 400   | 62.58% | 42.77% |
| 20.0%    | 46     | Neuro-KE     | pred       | 62    | 62.29% | 40.85% |
| 20.0%    | 46     | FeatOnly     | eeg        | 200   | 61.28% | 38.71% |
| 20.0%    | 46     | FeatOnly     | feat       | 200   | 61.46% | 39.90% |
| 20.0%    | 46     | FeatOnly     | full       | 400   | 63.23% | 41.12% |
| 20.0%    | 46     | FeatOnly     | pred       | 62    | 62.82% | 44.89% |
| 50.0%    | 116    | Baseline     | EEG        | 200   | 49.85% | 32.39% |
| 50.0%    | 116    | Neuro-KE     | eeg        | 200   | 66.94% | 46.95% |
| 50.0%    | 116    | Neuro-KE     | feat       | 200   | 67.83% | 45.46% |
| 50.0%    | 116    | Neuro-KE     | full       | 400   | 64.82% | 44.05% |
| 50.0%    | 116    | Neuro-KE     | pred       | 62    | 67.00% | 44.58% |
| 50.0%    | 116    | FeatOnly     | eeg        | 200   | 66.82% | 43.50% |
| 50.0%    | 116    | FeatOnly     | feat       | 200   | 67.35% | 43.74% |
| 50.0%    | 116    | FeatOnly     | full       | 400   | 65.29% | 46.13% |
| 50.0%    | 116    | FeatOnly     | pred       | 62    | 68.89% | 47.71% |
| 100%     | 232    | Baseline     | EEG        | 200   | 46.97% | 33.22% |
| 100%     | 232    | Neuro-KE     | eeg        | 200   | 65.47% | 47.68% |
| 100%     | 232    | Neuro-KE     | feat       | 200   | 67.24% | 47.12% |
| 100%     | 232    | Neuro-KE     | full       | 400   | 66.94% | 46.53% |
| 100%     | 232    | Neuro-KE     | pred       | 62    | 60.81% | 42.24% |
| 100%     | 232    | FeatOnly     | eeg        | 200   | 66.23% | 45.54% |
| 100%     | 232    | FeatOnly     | feat       | 200   | 66.41% | 45.87% |
| 100%     | 232    | FeatOnly     | full       | 400   | 66.41% | 48.18% |
| 100%     | 232    | FeatOnly     | pred       | 62    | 63.41% | 48.34% |
