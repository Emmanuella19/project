import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_date = pd.to_datetime("1966-01-01").date()
end_date = pd.to_datetime("2021-12-31").date()
mw_limit = 5.6
omni_file_name = "omni_daily.csv"
isc_file_name = "isc-gem-cat.csv"
var = "Density"

# -----------------------------
# Load and clean OMNI daily data
# -----------------------------
df = pd.read_csv(
    omni_file_name,
    sep=r"\s+",
    header=None,
    engine="python"
)

df = df.rename(columns={3: "Magnetic Field", 4: "Density", 5: "Velocity"})

bad_vals = [999.9, 9999.]
df = df[~df[["Magnetic Field", "Density", "Velocity"]].isin(bad_vals).any(axis=1)]
df = df.reset_index(drop=True)

df["date"] = pd.to_datetime(
    df[0].astype(str) + df[1].astype(str),
    format="%Y%j"
).dt.date

df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
df = df.sort_values("date").reset_index(drop=True)

print("OMNI data preview:")
print(df.head())

# -----------------------------
# Load ISC event data
# -----------------------------
cols = [
    "date","lat","lon","smajax","sminax","strike","q1","depth","unc1","q2",
    "mw","unc2","q3","s","mo","fac","mo_auth","mpp","mpr","mrr","mrt","mtp",
    "mtt","str1","dip1","rake1","str2","dip2","rake2","type","eventid"
]

isc_df = pd.read_csv(
    isc_file_name,
    sep=",",
    comment="#",
    skipinitialspace=True,
    header=None,
    names=cols,
    engine="python"
)

isc_df["date"] = pd.to_datetime(isc_df["date"].str.strip()).dt.date
isc_df["mw"] = pd.to_numeric(isc_df["mw"], errors="coerce")
isc_df["depth"] = pd.to_numeric(isc_df["depth"], errors="coerce")

isc_df = isc_df[["date", "mw", "depth"]]
isc_df = isc_df[isc_df["mw"] >= mw_limit]
isc_df = isc_df[(isc_df["date"] >= start_date) & (isc_df["date"] <= end_date)]

print(f"\nISC events with magnitude >= {mw_limit}: {len(isc_df)}")

# -----------------------------
# Keep only earthquakes on valid OMNI days
# -----------------------------
omni_dates = df["date"]
omni_dates_set = set(omni_dates)

isc_in_omni = isc_df[isc_df["date"].isin(omni_dates_set)].copy()

# Daily earthquake counts aligned with df rows
eq_counts_real = (
    isc_in_omni
    .groupby("date")
    .size()
    .reindex(omni_dates, fill_value=0)
    .to_numpy()
)

D = len(df)
E = eq_counts_real.sum()

print(f"\nTotal valid OMNI days (D): {D}")
print(f"Total ISC events on valid OMNI days (E): {E}")

# -----------------------------
# Prepare thresholds from original OMNI data
# -----------------------------
df_var = df[var]

V_min = df_var.min()
V_max = df_var.max()
V_av_ad = (df_var.mean() - V_min) / (V_max - V_min)
V_av_ad = round(V_av_ad, 2)

V_T_step = np.arange(V_av_ad, 0.55, 0.01)
V_T = V_min + V_T_step * (V_max - V_min)

print(f"\nVariable: {var}")
print(f"V_min = {V_min}")
print(f"V_max = {V_max}")
print(f"V_av_ad = {V_av_ad}")
print(f"Number of thresholds = {len(V_T)}")

# -----------------------------
# Precompute shifted OMNI values
# OMNI is NOT randomized in this test
# -----------------------------
today_vals = df[var]
tomorrow_vals = df[var].shift(-1)
yesterday_vals = df[var].shift(1)
day_after_tomorrow_vals = df[var].shift(-2)
day_before_yesterday_vals = df[var].shift(2)

# -----------------------------
# Helper function for R
# -----------------------------
def compute_R(eq_counts, mask):
    """
    eq_counts: daily earthquake counts aligned with df rows
    mask: boolean array selecting condition days
    """

    mask = np.asarray(mask, dtype=bool)

    DC = mask.sum()
    EC = eq_counts[mask].sum()

    if DC == 0 or (D - DC) == 0:
        return DC, EC, np.nan

    condition_rate = EC / DC
    non_condition_rate = (E - EC) / (D - DC)

    if non_condition_rate <= 0:
        return DC, EC, np.nan

    R = condition_rate / non_condition_rate

    return DC, EC, R

# -----------------------------
# First compute REAL R curve
# -----------------------------
real_results = []

for condition in range(0, 6):
    print(f"Processing REAL condition C{condition}...")

    for V_T_i in V_T:

        if condition == 0:
            condition_mask = today_vals < V_T_i

        elif condition == 1:
            condition_mask = today_vals >= V_T_i

        elif condition == 2:
            # second-to-last day above threshold:
            # tomorrow is still above, day after tomorrow is below
            condition_mask = (
                (today_vals >= V_T_i)
                & (tomorrow_vals >= V_T_i)
                & (day_after_tomorrow_vals < V_T_i)
            )

        elif condition == 3:
            # last day above threshold
            condition_mask = (
                (today_vals >= V_T_i)
                & (tomorrow_vals < V_T_i)
            )

        elif condition == 4:
            # first day below threshold
            # this is the paper's important "1Dy bT"-type condition
            condition_mask = (
                (today_vals < V_T_i)
                & (yesterday_vals >= V_T_i)
            )

        elif condition == 5:
            # second day below threshold
            condition_mask = (
                (today_vals < V_T_i)
                & (yesterday_vals < V_T_i)
                & (day_before_yesterday_vals >= V_T_i)
            )

        mask = condition_mask.fillna(False).to_numpy()

        DC, EC, R = compute_R(eq_counts_real, mask)

        real_results.append({
            "Seed": "REAL",
            "Condition": f"C{condition}",
            "V_T_step": round((V_T_i - V_min) / (V_max - V_min), 2),
            "V_T": round(V_T_i, 4),
            "DC": DC,
            "EC": EC,
            "R": R
        })

real_results_df = pd.DataFrame(real_results)

# -----------------------------
# RANDOMIZATION TEST 1
# Circularly shift earthquake daily counts
# -----------------------------
n_random = 100000   # use 10 while debugging, then 10000 or 100000
rng = np.random.default_rng(12345)

random_results = []

for run in range(n_random):

    print(f"Randomization run {run}/{n_random}")

    # Circularly shift earthquake counts by random number of days.
    # This preserves the sequence of earthquake counts but changes timing relative to OMNI.
    shift = rng.integers(1, D)
    eq_counts_rand = np.roll(eq_counts_real, shift)

    for condition in range(0, 6):

        for V_T_i in V_T:

            if condition == 0:
                condition_mask = today_vals < V_T_i

            elif condition == 1:
                condition_mask = today_vals >= V_T_i

            elif condition == 2:
                condition_mask = (
                    (today_vals >= V_T_i)
                    & (tomorrow_vals >= V_T_i)
                    & (day_after_tomorrow_vals < V_T_i)
                )

            elif condition == 3:
                condition_mask = (
                    (today_vals >= V_T_i)
                    & (tomorrow_vals < V_T_i)
                )

            elif condition == 4:
                condition_mask = (
                    (today_vals < V_T_i)
                    & (yesterday_vals >= V_T_i)
                )

            elif condition == 5:
                condition_mask = (
                    (today_vals < V_T_i)
                    & (yesterday_vals < V_T_i)
                    & (day_before_yesterday_vals >= V_T_i)
                )

            mask = condition_mask.fillna(False).to_numpy()

            DC, EC, R = compute_R(eq_counts_rand, mask)

            random_results.append({
                "Seed": run,
                "Shift": shift,
                "Condition": f"C{condition}",
                "V_T_step": round((V_T_i - V_min) / (V_max - V_min), 2),
                "V_T": round(V_T_i, 4),
                "DC": DC,
                "EC": EC,
                "R": R
            })

random_results_df = pd.DataFrame(random_results)

# -----------------------------
# Save raw results
# -----------------------------
real_results_df.to_csv(f"omni_threshold_results_{var}_REAL.csv", index=False)
random_results_df.to_csv(f"omni_threshold_results_{var}_circular_shift_{n_random}.csv", index=False)

print("\nREAL results preview:")
print(real_results_df.head())

print("\nRandomized results preview:")
print(random_results_df.head())

# -----------------------------
# Build pointwise random envelopes
# -----------------------------
summary_df = (
    random_results_df
    .groupby(["Condition", "V_T_step", "V_T"])["R"]
    .quantile([0.025, 0.05, 0.5, 0.95, 0.975, 0.99])
    .unstack()
    .reset_index()
)

summary_df = summary_df.rename(columns={
    0.025: "R_q025",
    0.05: "R_q05",
    0.5: "R_median",
    0.95: "R_q95",
    0.975: "R_q975",
    0.99: "R_q99"
})

summary_df.to_csv(f"omni_threshold_results_{var}_circular_shift_summary.csv", index=False)

# -----------------------------
# Pointwise p-values
# p = fraction of randomized R >= real R
# -----------------------------
pval_rows = []

for _, row in real_results_df.iterrows():

    cond = row["Condition"]
    step = row["V_T_step"]
    real_R = row["R"]

    subset = random_results_df[
        (random_results_df["Condition"] == cond)
        & (random_results_df["V_T_step"] == step)
    ]

    rand_R = subset["R"].dropna().to_numpy()

    if np.isfinite(real_R) and len(rand_R) > 0:
        # one-sided test for unusually high R
        p_one_sided = np.mean(rand_R >= real_R)

        # two-sided style test around R=1, useful if you care about high or low anomalies
        p_two_sided = np.mean(np.abs(rand_R - 1) >= abs(real_R - 1))
    else:
        p_one_sided = np.nan
        p_two_sided = np.nan

    pval_rows.append({
        "Condition": cond,
        "V_T_step": step,
        "V_T": row["V_T"],
        "R_real": real_R,
        "p_one_sided_high_R": p_one_sided,
        "p_two_sided_away_from_1": p_two_sided
    })

pval_df = pd.DataFrame(pval_rows)
pval_df.to_csv(f"omni_threshold_results_{var}_circular_shift_pvalues.csv", index=False)

print("\nPointwise p-value preview:")
print(pval_df.head())

# -----------------------------
# Global curve test: max R over thresholds for each condition
# -----------------------------
global_rows = []

for condition in sorted(real_results_df["Condition"].unique()):

    real_cond = real_results_df[real_results_df["Condition"] == condition]
    real_max_R = real_cond["R"].max()

    rand_max_R = (
        random_results_df[random_results_df["Condition"] == condition]
        .groupby("Seed")["R"]
        .max()
        .dropna()
        .to_numpy()
    )

    p_global = np.mean(rand_max_R >= real_max_R)

    global_rows.append({
        "Condition": condition,
        "real_max_R": real_max_R,
        "random_median_max_R": np.median(rand_max_R),
        "random_95pct_max_R": np.quantile(rand_max_R, 0.95),
        "random_99pct_max_R": np.quantile(rand_max_R, 0.99),
        "p_global_max_R": p_global
    })

global_df = pd.DataFrame(global_rows)
global_df.to_csv(f"omni_threshold_results_{var}_circular_shift_global_test.csv", index=False)

print("\nGlobal max-R test:")
print(global_df)

# -----------------------------
# Plot real curve against random envelope
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)

for i, condition in enumerate(range(0, 6)):

    ax = axes[i // 3, i % 3]

    cond_label = f"C{condition}"

    real_sub = real_results_df[real_results_df["Condition"] == cond_label]
    sum_sub = summary_df[summary_df["Condition"] == cond_label]

    ax.fill_between(
        sum_sub["V_T_step"],
        sum_sub["R_q025"],
        sum_sub["R_q975"],
        alpha=0.25,
        label="95% random envelope"
    )

    ax.plot(
        sum_sub["V_T_step"],
        sum_sub["R_median"],
        linestyle="--",
        label="random median"
    )

    ax.plot(
        real_sub["V_T_step"],
        real_sub["R"],
        linewidth=2,
        label="real"
    )

    ax.axhline(1, linestyle=":", linewidth=1)

    ax.set_title(cond_label)
    ax.set_xlabel("V_T_step")
    ax.set_ylabel("R")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"omni_threshold_results_{var}_circular_shift_envelope_{n_random}.png", dpi=200)
plt.show()