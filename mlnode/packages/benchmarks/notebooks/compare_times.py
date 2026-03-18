import numpy as np
from scipy import stats

def load_data(filepath):
    with open(filepath, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

new_data = load_data('benchmark_val_new_H.txt')
old_data = load_data('benchmark_val_old_H.txt')

print(f"{'':=<60}")
print(f"  Dataset comparison: new_H vs old_H")
print(f"{'':=<60}\n")

print(f"{'Metric':<30} {'New H':>14} {'Old H':>14}")
print(f"{'-'*58}")
print(f"{'N (sample size)':<30} {len(new_data):>14d} {len(old_data):>14d}")
print(f"{'Mean (E[X])':<30} {np.mean(new_data):>14.3f} {np.mean(old_data):>14.3f}")
print(f"{'Std Dev (σ, population)':<30} {np.std(new_data):>14.3f} {np.std(old_data):>14.3f}")
print(f"{'Std Dev (s, sample)':<30} {np.std(new_data, ddof=1):>14.3f} {np.std(old_data, ddof=1):>14.3f}")
print(f"{'Median':<30} {np.median(new_data):>14.3f} {np.median(old_data):>14.3f}")
print(f"{'Min':<30} {np.min(new_data):>14.3f} {np.min(old_data):>14.3f}")
print(f"{'Max':<30} {np.max(new_data):>14.3f} {np.max(old_data):>14.3f}")

print(f"\n{'':=<60}")
print(f"  Statistical tests")
print(f"{'':=<60}\n")

# Welch's t-test (does not assume equal variances)
t_stat, t_pval = stats.ttest_ind(new_data, old_data, equal_var=False)
print(f"Welch's t-test:")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value     = {t_pval:.4e}")
print(f"  Significant at α=0.05? {'Yes' if t_pval < 0.05 else 'No'}\n")

# Mann-Whitney U test (non-parametric)
u_stat, u_pval = stats.mannwhitneyu(new_data, old_data, alternative='two-sided')
print(f"Mann-Whitney U test:")
print(f"  U-statistic = {u_stat:.4f}")
print(f"  p-value     = {u_pval:.4e}")
print(f"  Significant at α=0.05? {'Yes' if u_pval < 0.05 else 'No'}\n")

# Kolmogorov-Smirnov test
ks_stat, ks_pval = stats.ks_2samp(new_data, old_data)
print(f"Kolmogorov-Smirnov test:")
print(f"  KS-statistic = {ks_stat:.4f}")
print(f"  p-value      = {ks_pval:.4e}")
print(f"  Significant at α=0.05? {'Yes' if ks_pval < 0.05 else 'No'}\n")

# Effect size (Cohen's d)
pooled_std = np.sqrt((np.std(new_data, ddof=1)**2 + np.std(old_data, ddof=1)**2) / 2)
cohens_d = (np.mean(new_data) - np.mean(old_data)) / pooled_std
print(f"Cohen's d (effect size) = {cohens_d:.4f}")
magnitude = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
print(f"  Magnitude: {magnitude}")
