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
print("OMNI data preview before randomization:")
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
# Compute D and E
# -----------------------------
D = len(df)
omni_dates = set(df["date"])
isc_in_omni = isc_df[isc_df["date"].isin(omni_dates)]
E = len(isc_in_omni)

print(f"\nTotal valid OMNI days (D): {D}")
print(f"Total ISC events on valid OMNI days (E): {E}")

# -----------------------------
# Prepare thresholds
# -----------------------------
df_var_original = df[var]
V_av_ad = (df_var_original.mean() - df_var_original.min()) / (df_var_original.max() - df_var_original.min())
V_av_ad = round(V_av_ad, 2)

V_T_step = np.arange(V_av_ad, 0.55, 0.01)
V_T = df_var_original.min() + V_T_step * (df_var_original.max() - df_var_original.min())

# -----------------------------
# RANDOMIZATION LOOP (10 runs)
# -----------------------------
results = []

for seed in range(10):
    print(f"\n=== Randomization run {seed} ===")

    # Shuffle OMNI values
    df_randomized = df.copy()
    df_randomized[["Magnetic Field", "Density", "Velocity"]] = (
        df_randomized[["Magnetic Field", "Density", "Velocity"]]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    # Use randomized variable
    df_var = df_randomized[var]

    # Precompute shifted values from randomized data
    tomorrow_vals = df_randomized[var].shift(-1)
    yesterday_vals = df_randomized[var].shift(1)
    day_after_tomorrow_vals = df_randomized[var].shift(-2)
    day_before_yesterday_vals = df_randomized[var].shift(2)

    # -----------------------------
    # Loop over all thresholds
    # -----------------------------
    for condition in range(0, 6):
        print(f"Processing condition C{condition}...")

        for V_T_i in V_T:

            V_T_step_i = (V_T_i - df_var.min()) / (df_var.max() - df_var.min())

            # Conditions using df_randomized
            if condition == 0:
                condition_mask = (df_randomized[var] < V_T_i)
            if condition == 1:
                condition_mask = (df_randomized[var] >= V_T_i)
            if condition == 2:
                condition_mask = (day_after_tomorrow_vals < V_T_i) & (tomorrow_vals >= V_T_i)
            if condition == 3:
                condition_mask = (df_randomized[var] >= V_T_i) & (tomorrow_vals < V_T_i)
            if condition == 4:
                condition_mask = (df_randomized[var] < V_T_i) & (yesterday_vals >= V_T_i)
            if condition == 5:
                condition_mask = (df_randomized[var] < V_T_i) & (yesterday_vals < V_T_i) & (day_before_yesterday_vals >= V_T_i)

            condition_dates = set(df_randomized.loc[condition_mask, "date"])
            DC = len(condition_dates)
            EC = len(isc_in_omni[isc_in_omni["date"].isin(condition_dates)])

            if DC > 0 and (D - DC) > 0:
                non_condition_rate = (E - EC) / (D - DC)
                R = (EC / DC) / non_condition_rate if non_condition_rate != 0 else np.nan
            else:
                R = np.nan

            results.append({
                "Seed": seed,
                "Condition": f"C{condition}",
                "V_T_step": round(V_T_step_i, 2),
                "V_T": round(V_T_i, 4),
                "DC": DC,
                "EC": EC,
                "R": R
            })

# -----------------------------
# Save results
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"omni_threshold_results_{var}_10randomizations.csv", index=False)

print("\nResults preview:")
print(results_df.head(10))

# -----------------------------
# Plot (optional)
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, condition in enumerate(range(0, 6)):
    ax = axes[i // 3, i % 3]
    condition_results = results_df[results_df["Condition"] == f"C{condition}"]
    for seed in range(10):
        subset = condition_results[condition_results["Seed"] == seed]
        ax.plot(subset["V_T_step"], subset["R"], alpha=0.4)
    ax.set_title(f"C{condition}")
    ax.set_xlabel("V_T_step")
    ax.set_ylabel("R")
    ax.grid()

plt.tight_layout()
plt.savefig(f"omni_threshold_results_{var}_10randomizations.png")
plt.show()
