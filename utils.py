import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_stats(s):
    """
    Beräknar grundläggande beskrivande statistik för en pandas Series.

    Parametrar
    s : pandas.Series

    Returnerar
        Dictionary med mean, median, min och max.
    """
    arr = np.array(s.dropna(), dtype=float)
    return {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'min': np.min(arr),
        'max': np.max(arr)
    }

def fit_bp_regression(df):
    """
    Tränar en linjär regressionsmodell för blodtryck
    baserat på ålder och vikt.

    Returnerar modell, data och förutsägelse.
    """
    data = df[["age", "weight", "systolic_bp"]].dropna()

    X = data[["age", "weight"]].to_numpy()
    y = data["systolic_bp"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    return model, data, X, y, y_pred

def plot_bp_vs_age(model, data, y):
    """
    Visualiserar sambandet mellan ålder och systoliskt blodtryck
    med vikten fixerad till sitt medelvärde.

    Parametrar
    model : sklearn LinearRegression tränad regressionsmodell.

    data : DataFrame som innehåller kolumnerna 'age' och 'weight'.

    y : Observerade blodtrycksvärden.
    """
    mean_weight = data["weight"].mean()
    age_line = np.linspace(data["age"].min(), data["age"].max(), 100)

    X_line = np.column_stack([
        age_line,
        np.full_like(age_line, mean_weight)
    ])

    y_line = model.predict(X_line)

    plt.figure(figsize=(7, 3))
    plt.scatter(data["age"], y, alpha=0.4, label="Data")
    plt.plot(age_line, y_line, color="red", label="Regression (vikt = medel)")
    plt.xlabel("Ålder")
    plt.ylabel("Systoliskt blodtryck")
    plt.title("Blodtryck jämfört med ålder (vikt fixerad)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bp_vs_weight(model, data, y):
    """
    Visualiserar sambandet mellan vikt och systoliskt blodtryck
    med åldern fixerad till sitt medelvärde.

    Parametrar
    model : sklearn LinearRegression tränad regressionsmodell.

    data : DataFrame som innehåller kolumnerna 'age' och 'weight'.

    y : Observerade blodtrycksvärden.
    """
    mean_age = data["age"].mean()
    weight_line = np.linspace(data["weight"].min(), data["weight"].max(), 100)

    X_line = np.column_stack([
        np.full_like(weight_line, mean_age),
        weight_line
    ])

    y_line = model.predict(X_line)

    plt.figure(figsize=(7, 3))
    plt.scatter(data["weight"], y, alpha=0.4, label="Data")
    plt.plot(weight_line, y_line, color="red", label="Regression (ålder = medel)")
    plt.xlabel("Vikt")
    plt.ylabel("Systoliskt blodtryck")
    plt.title("Blodtryck jämfört med vikt (ålder fixerad)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_basic_stats(df):
    """
    Beräknar grundläggande statistik (mean, median, min, max)
    för utvalda variabler i datasetet.

    Parametrar
    df : Dataset med kolumner för ålder, vikt, längd, blodtryck och kolesterol.

    Returnerar
        Dictionary där varje variabel har en dictionary med statistik.
    """
    variables = {
        "Age": df["age"],
        "Weight": df["weight"],
        "Height": df["height"],
        "Systolic BP": df["systolic_bp"],
        "Cholesterol": df["cholesterol"]
    }

    stats_summary = {}

    for name, series in variables.items():
        arr = np.array(series.dropna(), dtype=float)
        stats_summary[name] = {
            "mean": np.mean(arr),
            "median": np.median(arr),
            "min": np.min(arr),
            "max": np.max(arr)
        }

    return stats_summary

def plot_bp_histogram(bp_series, stats):
    """
    Plottar ett histogram över blodtryck och markerar medelvärdet.

    Parametrar
    bp_series : Blodtrycksvärden.
    stats : Dictionary med grundläggande statistik (innehåller 'mean').
    """
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.hist(bp_series, bins=30, edgecolor="black")
    ax.axvline(stats["mean"], color="tab:green", linestyle="--", label="Medelvärde")

    ax.set_title("Histogram över blodtryck")
    ax.set_xlabel("Blodtryck")
    ax.set_ylabel("Antal")
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.show()

def plot_weight_boxplot_by_sex(df, sex):
    """
    Plottar en boxplot för vikt uppdelat på kön.

    Parametrar
    df: Dataset som innehåller kolumnerna 'sex' och 'weight'.
    sex : str -> 'M' för män eller 'F' för kvinnor.
    """
    weights = df.loc[df["sex"] == sex, "weight"].dropna().to_numpy()

    title_map = {"M": "Vikt (Män)", "F": "Vikt (Kvinnor)"}
    title = title_map.get(sex, "Vikt")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.boxplot(weights, vert=False)
    ax.set_title(title)
    ax.set_xlabel("Vikt")
    ax.grid(axis="x")

    plt.tight_layout()
    plt.show()

def compute_smoking_counts(df):
    """
    Beräknar antal rökare och icke-rökare uppdelat på kön samt totalt.

    Parametrar
    df : Dataset som innehåller kolumnerna 'sex' och 'smoker'.

    Returnerar
    labels : Etiketter för stapeldiagrammet.
    values : Antal individer i respektive grupp.
    colors : Färger för respektive stapel.
    """
    female = df.loc[df["sex"] == "F", "smoker"]
    male = df.loc[df["sex"] == "M", "smoker"]

    female_non = np.sum(female == "No")
    female_yes = np.sum(female == "Yes")
    male_non = np.sum(male == "No")
    male_yes = np.sum(male == "Yes")

    total_non = female_non + male_non
    total_yes = female_yes + male_yes

    labels = [
        "Kvinnor\nicke rökare",
        "Kvinnor\nrökare",
        "Män\nicke rökare",
        "Män\nrökare",
        "Totalt\nicke rökare",
        "Totalt\nrökare"
    ]

    values = [
        female_non, female_yes,
        male_non, male_yes,
        total_non, total_yes
    ]

    colors = [
        "lightcoral", "lightcoral",
        "skyblue", "skyblue",
        "gray", "gray"
    ]

    return labels, values, colors

def plot_smoking_by_sex(labels, values, colors):
    """
    Plottar stapeldiagram över rökning uppdelat på kön och totalt.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(labels, values, color=colors)

    ax.set_title("Rökning")
    ax.set_ylabel("Antal")
    ax.set_xlabel("Grupper")
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.show()

def simulate_disease_prevalence(df, n_sim=1000):
    """
    Simulerar sjukdomsförekomst baserat på observerad andel i datan.

    Parametrar
    df : Dataset som innehåller kolumnen 'disease' (0/1).
    n_sim : Antal individer i simuleringen.

    Returnerar
        Dictionary med observerade värden, simulering och spridningsmått.
    """
    people_with_disease = df.loc[df["disease"] == 1]
    people_without_disease = df.loc[df["disease"] == 0]

    true_disease_chance = len(people_with_disease) / (
        len(people_with_disease) + len(people_without_disease)
    )

    simulation = np.random.binomial(n=1, p=true_disease_chance, size=n_sim)
    simulated_cases = simulation.sum()

    std = np.sqrt(n_sim * true_disease_chance * (1 - true_disease_chance))

    results = {
        "n_with_disease": len(people_with_disease),
        "n_total": len(df),
        "true_rate": true_disease_chance,
        "simulated_cases": simulated_cases,
        "simulated_rate": simulated_cases / n_sim,
        "expected_cases": true_disease_chance * n_sim,
        "std": std,
        "normal_interval": (
            true_disease_chance * n_sim - std,
            true_disease_chance * n_sim + std
        )
    }

    return results

def compute_disease_prevalence_by_sex(df):
    """
    Beräknar andelen individer med sjukdom uppdelat på kön.

    Parametrar
    df : Dataset som innehåller kolumnerna 'sex' och 'disease' (0/1).

    Returnerar
    labels : Namn på grupperna.
    values : Andel individer med sjukdom per grupp.
    colors : Färger för stapeldiagrammet.
    """
    disease_by_sex = df.groupby("sex")["disease"].mean()

    labels = ["Kvinnor", "Män"]
    values = disease_by_sex.loc[["F", "M"]].values.tolist()
    colors = ["lightcoral", "skyblue"]

    return labels, values, colors

def plot_disease_prevalence_by_sex(labels, values, colors):
    """
    Plottar sjukdomsförekomst uppdelat på kön.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(labels, values, color=colors)

    ax.set_ylabel("Andel med sjukdom")
    ax.set_xlabel("Kön")
    ax.set_title("Sjukdomsförekomst per kön")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.show()

def bootstrap_bp_difference_by_smoking(df, n_boot=10_000):
    """
    Utför ett bootstrap-test för skillnaden i medelvärde av systoliskt blodtryck
    mellan rökare och icke-rökare.

    Parametrar
    df : Dataset som innehåller kolumnerna 'smoker' och 'systolic_bp'.
    n_boot : Antal bootstrap-resamplings.

    Returnerar
    Dictionary med observerad skillnad, p-värde och konfidensintervall.
    """
    bp_smoker = df.loc[df["smoker"] == "Yes", "systolic_bp"].to_numpy()
    bp_non_smoker = df.loc[df["smoker"] == "No", "systolic_bp"].to_numpy()

    obs_diff = bp_smoker.mean() - bp_non_smoker.mean()

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        smoker_sample = np.random.choice(bp_smoker, size=len(bp_smoker), replace=True)
        non_smoker_sample = np.random.choice(
            bp_non_smoker, size=len(bp_non_smoker), replace=True
        )
        boot_diffs[i] = smoker_sample.mean() - non_smoker_sample.mean()

    p_boot = np.mean(boot_diffs <= 0)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    results = {
        "obs_diff": obs_diff,
        "p_value": p_boot,
        "ci": (float(ci_low), float(ci_high)),
        "boot_diffs": boot_diffs
    }

    return results