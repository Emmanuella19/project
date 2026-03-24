import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_date = pd.to_datetime("1966-01-01").date()
end_date = pd.to_datetime("2021-12-31").date()
mw_limit = 5.6  # Minimum magnitude for ISC events to consider
omni_file_name = "omni_daily.csv"
isc_file_name = "isc-gem-cat.csv"
var = "Density"  # Change to "Magnetic Field" or "Velocity" or "Density" as needed

# -----------------------------
# Load and clean OMNI daily data
# -----------------------------
df = pd.read_csv(
    omni_file_name,
    sep=r"\s+",
    header=None,
    engine="python"
)

# Rename columns
df = df.rename(columns={3: "Magnetic Field", 4: "Density", 5: "Velocity"})

# Values that indicate bad data
bad_vals = [999.9, 9999.]

# Remove rows where Magnetic Field, Density, or Velocity has bad values
df = df[~df[["Magnetic Field", "Density", "Velocity"]].isin(bad_vals).any(axis=1)]

# Reset index after filtering
df = df.reset_index(drop=True)

# Create date column from year + day-of-year
df["date"] = pd.to_datetime(
    df[0].astype(str) + df[1].astype(str),
    format="%Y%j"
).dt.date

# Filter data to the specified date range
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

print("OMNI data preview:")
print(df.head())

df_var = df[var]

print(f"\nVariable: {var}")
print("Min:", df_var.min())
print("Max:", df_var.max())
print("Avg:", df_var.mean())

# -----------------------------
# Compute normalized V(av-ad)
# -----------------------------
V_av_ad = (df_var.mean() - df_var.min()) / (df_var.max() - df_var.min())
V_av_ad = round(V_av_ad, 2)

print("\nV(av-ad) =", V_av_ad)

# Create threshold values from V(av-ad) to 1.00
V_T_step = np.arange(V_av_ad, 0.55, 0.01)
V_T = df_var.min() + V_T_step * (df_var.max() - df_var.min())

print("\nVarying threshold values V_T:")
print(V_T)

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
print(isc_df.head())

# Filter according to magnitude
isc_df = isc_df[isc_df["mw"] >= mw_limit]

# Set date range for ISC events
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
# Loop over all thresholds (FAST)
# -----------------------------
results = []

#test comment

# Precompute tomorrow's values once
tomorrow_vals = df[var].shift(-1)

# Precompute yesterday's values once
yesterday_vals = df[var].shift(1)

# Precompute day after tomorrow's values once
day_after_tomorrow_vals = df[var].shift(-2)

# Precompute day before yesterday's values once
day_before_yesterday_vals = df[var].shift(2)

for condition in range(0, 6):
    print(f"\nProcessing condition C{condition}...")

    for V_T_i in V_T:

        # Corresponding V_T_step_i for reference
        V_T_step_i = (V_T_i - df_var.min()) / (df_var.max() - df_var.min())

        # Condition C0:
        # All the days with V below the VT threshold
        if condition == 0:
            condition_mask = (df[var] < V_T_i)

        # Condition C1:
        # All the days with V above the VT threshold
        if condition == 1:
            condition_mask = (df[var] >= V_T_i)

        # Condition C2:
        # 2nd to last day with V above the VT threshold
        if condition == 2:
            condition_mask = (day_after_tomorrow_vals < V_T_i) & (tomorrow_vals >= V_T_i)

        # Condition C3:
        # Last day with V above the VT threshold
        if condition == 3:
            condition_mask = (df[var] >= V_T_i) & (tomorrow_vals < V_T_i)

        # Condition C4:
        # 1st day with V below the VT threshold
        if condition == 4:
            condition_mask = (df[var] < V_T_i) & (yesterday_vals >= V_T_i)


        # Condition C5:
        # 2nd day with V below the VT threshold
        if condition == 5:
            condition_mask = (df[var] < V_T_i) & (yesterday_vals < V_T_i) & (day_before_yesterday_vals >= V_T_i)

        # Condition dates
        condition_dates = set(df.loc[condition_mask, "date"])

        # DC = number of days satisfying condition
        DC = len(condition_dates)

        # EC = number of events occurring on those days
        EC = len(isc_in_omni[isc_in_omni["date"].isin(condition_dates)])

        # Compute R
        if DC > 0 and (D - DC) > 0:
            non_condition_event_rate = (E - EC) / (D - DC)

            if non_condition_event_rate != 0:
                R = (EC / DC) / non_condition_event_rate
            else:
                R = np.nan
        else:
            R = np.nan

        results.append({
            "Condition": f"C{condition}",
            "V_T_step": round(V_T_step_i, 2),
            "V_T": round(V_T_i, 4),
            "DC": DC,
            "EC": EC,
            "D": D,
            "E": E,
            "R": R
        })

# -----------------------------
# Save results
# -----------------------------
results_df = pd.DataFrame(results)

print("\nResults preview:")
print(results_df.head(10))

# Save to CSV
results_df.to_csv(f"omni_threshold_results_{var}.csv", index=False)

#plot for each condition in separate subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, condition in enumerate(range(0, 6)):
    ax = axes[i // 3, i % 3]
    condition_results = results_df[results_df["Condition"] == f"C{condition}"]
    ax.plot(condition_results["V_T_step"], condition_results["R"], marker='o')
    if condition == 0:
        ax.set_title("C0: All the days with V below the VT threshold")
    if condition == 1:
        ax.set_title("C1: All the days with V above the VT threshold")
    if condition == 2:
        ax.set_title("C2: 2nd to last day with V above the VT threshold")
    if condition == 3:
        ax.set_title("C3: Last day with V above the VT threshold")
    if condition == 4:
        ax.set_title("C4: 1st day with V below the VT threshold")
    if condition == 5:
        ax.set_title("C5: 2nd day with V below the VT threshold")
    ax.set_xlabel("V_T_step")
    ax.set_ylabel("R")
    ax.grid()

plt.tight_layout()
plt.savefig(f"omni_threshold_results_{var}.png")
plt.show()