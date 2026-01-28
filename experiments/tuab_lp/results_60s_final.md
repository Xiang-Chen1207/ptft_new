Loading Baseline features from experiments/tuab_lp/features/recon_features.npz...
Loading Flagship features from experiments/tuab_lp/features/neuro_ke_pooled_features.npz...
Loading Flagship Pooled features from experiments/tuab_lp/features/neuro_ke_pooled_features.npz...
Loading FeatOnly features from experiments/tuab_lp_feat_only/features/feat_only_features.npz...
Total Unique Training Subjects: 1851

=== Comparative Results Table ===
| Ratio    | NumSub | Model        | Feature    | Dim   | Acc     | BAcc    |
|----------|--------|--------------|------------|-------|---------|---------|
| 0.5%     | 9      | Baseline     | EEG        | 200   | 61.82% | 61.64% |
| 0.5%     | 9      | Flagship(Full) | eeg        | 200   | 63.03% | 62.79% |
| 0.5%     | 9      | Flagship(Full) | feat       | 200   | 60.72% | 60.70% |
| 0.5%     | 9      | Flagship(Full) | full       | 400   | 58.92% | 58.70% |
| 0.5%     | 9      | Flagship(Pool) | feat       | 200   | 59.90% | 59.79% |
| 0.5%     | 9      | Flagship(Pool) | full       | 400   | 60.20% | 60.02% |
| 0.5%     | 9      | FeatOnly     | eeg        | 200   | 54.87% | 55.03% |
| 0.5%     | 9      | FeatOnly     | feat       | 200   | 57.10% | 57.21% |
| 0.5%     | 9      | FeatOnly     | full       | 400   | 57.24% | 57.38% |
| 0.5%     | 9      | FeatOnly     | pred       | 62    | 64.04% | 64.07% |
| 1.0%     | 18     | Baseline     | EEG        | 200   | 64.60% | 64.93% |
| 1.0%     | 18     | Flagship(Full) | eeg        | 200   | 68.05% | 68.31% |
| 1.0%     | 18     | Flagship(Full) | feat       | 200   | 67.59% | 67.92% |
| 1.0%     | 18     | Flagship(Full) | full       | 400   | 67.30% | 67.62% |
| 1.0%     | 18     | Flagship(Pool) | feat       | 200   | 66.67% | 67.01% |
| 1.0%     | 18     | Flagship(Pool) | full       | 400   | 67.81% | 68.13% |
| 1.0%     | 18     | FeatOnly     | eeg        | 200   | 64.14% | 64.61% |
| 1.0%     | 18     | FeatOnly     | feat       | 200   | 64.23% | 64.75% |
| 1.0%     | 18     | FeatOnly     | full       | 400   | 64.63% | 65.10% |
| 1.0%     | 18     | FeatOnly     | pred       | 62    | 62.80% | 63.30% |
| 5.0%     | 92     | Baseline     | EEG        | 200   | 68.23% | 68.21% |
| 5.0%     | 92     | Flagship(Full) | eeg        | 200   | 75.21% | 75.30% |
| 5.0%     | 92     | Flagship(Full) | feat       | 200   | 74.60% | 74.68% |
| 5.0%     | 92     | Flagship(Full) | full       | 400   | 74.50% | 74.58% |
| 5.0%     | 92     | Flagship(Pool) | feat       | 200   | 74.53% | 74.61% |
| 5.0%     | 92     | Flagship(Pool) | full       | 400   | 74.35% | 74.43% |
| 5.0%     | 92     | FeatOnly     | eeg        | 200   | 72.26% | 72.25% |
| 5.0%     | 92     | FeatOnly     | feat       | 200   | 71.81% | 71.79% |
| 5.0%     | 92     | FeatOnly     | full       | 400   | 72.39% | 72.38% |
| 5.0%     | 92     | FeatOnly     | pred       | 62    | 70.28% | 70.23% |
| 10.0%    | 185    | Baseline     | EEG        | 200   | 69.69% | 69.65% |
| 10.0%    | 185    | Flagship(Full) | eeg        | 200   | 76.98% | 77.07% |
| 10.0%    | 185    | Flagship(Full) | feat       | 200   | 76.22% | 76.31% |
| 10.0%    | 185    | Flagship(Full) | full       | 400   | 76.12% | 76.20% |
| 10.0%    | 185    | Flagship(Pool) | feat       | 200   | 76.28% | 76.37% |
| 10.0%    | 185    | Flagship(Pool) | full       | 400   | 76.21% | 76.28% |
| 10.0%    | 185    | FeatOnly     | eeg        | 200   | 75.37% | 75.40% |
| 10.0%    | 185    | FeatOnly     | feat       | 200   | 74.14% | 74.18% |
| 10.0%    | 185    | FeatOnly     | full       | 400   | 74.77% | 74.82% |
| 10.0%    | 185    | FeatOnly     | pred       | 62    | 72.12% | 72.11% |
| 20.0%    | 370    | Baseline     | EEG        | 200   | 69.01% | 68.80% |
| 20.0%    | 370    | Flagship(Full) | eeg        | 200   | 76.68% | 76.65% |
| 20.0%    | 370    | Flagship(Full) | feat       | 200   | 76.08% | 76.05% |
| 20.0%    | 370    | Flagship(Full) | full       | 400   | 76.44% | 76.41% |
| 20.0%    | 370    | Flagship(Pool) | feat       | 200   | 76.19% | 76.15% |
| 20.0%    | 370    | Flagship(Pool) | full       | 400   | 76.31% | 76.28% |
| 20.0%    | 370    | FeatOnly     | eeg        | 200   | 75.45% | 75.36% |
| 20.0%    | 370    | FeatOnly     | feat       | 200   | 74.30% | 74.19% |
| 20.0%    | 370    | FeatOnly     | full       | 400   | 75.46% | 75.39% |
| 20.0%    | 370    | FeatOnly     | pred       | 62    | 72.85% | 72.71% |
| 50.0%    | 925    | Baseline     | EEG        | 200   | 69.34% | 69.16% |
| 50.0%    | 925    | Flagship(Full) | eeg        | 200   | 76.60% | 76.53% |
| 50.0%    | 925    | Flagship(Full) | feat       | 200   | 76.23% | 76.15% |
| 50.0%    | 925    | Flagship(Full) | full       | 400   | 76.71% | 76.64% |
| 50.0%    | 925    | Flagship(Pool) | feat       | 200   | 76.37% | 76.28% |
| 50.0%    | 925    | Flagship(Pool) | full       | 400   | 76.68% | 76.61% |
| 50.0%    | 925    | FeatOnly     | eeg        | 200   | 75.54% | 75.42% |
| 50.0%    | 925    | FeatOnly     | feat       | 200   | 75.14% | 75.01% |
| 50.0%    | 925    | FeatOnly     | full       | 400   | 75.63% | 75.52% |
| 50.0%    | 925    | FeatOnly     | pred       | 62    | 73.43% | 73.28% |
| 100%     | 1851   | Baseline     | EEG        | 200   | 70.00% | 69.97% |
| 100%     | 1851   | Flagship(Full) | eeg        | 200   | 77.60% | 77.60% |
| 100%     | 1851   | Flagship(Full) | feat       | 200   | 77.41% | 77.41% |
| 100%     | 1851   | Flagship(Full) | full       | 400   | 77.81% | 77.81% |
| 100%     | 1851   | Flagship(Pool) | feat       | 200   | 77.45% | 77.45% |
| 100%     | 1851   | Flagship(Pool) | full       | 400   | 77.84% | 77.84% |
| 100%     | 1851   | FeatOnly     | eeg        | 200   | 76.70% | 76.68% |
| 100%     | 1851   | FeatOnly     | feat       | 200   | 76.30% | 76.27% |
| 100%     | 1851   | FeatOnly     | full       | 400   | 76.83% | 76.82% |
| 100%     | 1851   | FeatOnly     | pred       | 62    | 74.27% | 74.23% |
