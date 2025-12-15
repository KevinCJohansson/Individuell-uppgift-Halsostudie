import numpy as np
from math import sqrt

def get_stats(s):
    arr = np.array(s.dropna(), dtype=float)
    return {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'min': np.min(arr),
        'max': np.max(arr)
    }

class HealthAnalyzer:
    def __init__(self, df):
        self.df = df
    
    def ci_mean_normal(self, x):
        x = np.asarray(x, dtype=float)
        mean_x = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        n = len(x)

        z_critical = 1.96
        half_width = z_critical * s / sqrt(n)
        lo, hi = mean_x - half_width, mean_x + half_width
        return lo, hi, mean_x, s, n
    
    def ci_mean_bootstrap(self, x, B=5000, confidence=0.95):
        x = np.asarray(x, dtype=float)
        n = len(x)
        boot_means = np.empty(B)
        
        for b in range(B):
            boot_sample = np.random.choice(x, size=n, replace=True)
            boot_means[b] = np.mean(boot_sample)

        alpha = (1 - confidence) / 2
        lo, hi = np.percentile(boot_means, [100*alpha, 100*(1 - alpha)])
        return float(lo), float(hi), float(np.mean(x)), boot_means
    
    def covers_true_mean(self, x, true_mean, method = "normal", n=40, trials=200):
        hits = 0
        for _ in range(trials):
            trial_sample = np.random.choice(x, size=n, replace=True)
            if method == "normal":
                lo, hi, *_ = self.ci_mean_normal(trial_sample)
            else:
                lo, hi, *_ = self.ci_mean_bootstrap(trial_sample, B=1500)

            hits += (lo <= true_mean <= hi)

        return hits / trials