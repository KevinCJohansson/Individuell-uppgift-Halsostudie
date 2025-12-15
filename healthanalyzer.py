import numpy as np
from math import sqrt

class HealthAnalyzer:
    """Analyserar hälsodata med konfidensintervall och simuleringar."""

    def __init__(self, df):
        """
        Parametrar
        df : Datasetet som analysen baseras på.
        """
        self.df = df

    def ci_mean_normal(self, x, confidence=0.95):
        """
        Beräknar konfidensintervall för medelvärdet med normalapproximation.

        Parametrar
        x : Stickprov av numeriska värden.
        confidence : Konfidensnivå (t.ex. 0.95).

        Returnerar
        lo, hi, mean_x, s, n
        """
        x = np.asarray(x, dtype=float)
        mean_x = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        n = len(x)

        z_critical = 1.96
        half_width = z_critical * s / sqrt(n)
        lo, hi = mean_x - half_width, mean_x + half_width
        return lo, hi, mean_x, s, n

    def ci_mean_bootstrap(self, x, B=5000, confidence=0.95):
        """
        Beräknar konfidensintervall för medelvärdet med bootstrap.

        Returnerar (lo, hi, mean_x, boot_means).
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        boot_means = np.empty(B)

        for b in range(B):
            boot_sample = np.random.choice(x, size=n, replace=True)
            boot_means[b] = np.mean(boot_sample)

        alpha = (1 - confidence) / 2
        lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
        return float(lo), float(hi), float(np.mean(x)), boot_means

    def covers_true_mean(self, x, true_mean, method="normal", n=40, trials=200, B=1500):
        """
        Simulerar hur ofta ett CI täcker det sanna medelvärdet.

        method: "normal" eller "bootstrap"
        """
        x = np.asarray(x, dtype=float)

        hits = 0
        for _ in range(trials):
            trial_sample = np.random.choice(x, size=n, replace=True)

            if method == "normal":
                lo, hi, *_ = self.ci_mean_normal(trial_sample)
            elif method == "bootstrap":
                lo, hi, *_ = self.ci_mean_bootstrap(trial_sample, B=B)
            else:
                raise ValueError("method must be 'normal' or 'bootstrap'")

            hits += (lo <= true_mean <= hi)

        return hits / trials
