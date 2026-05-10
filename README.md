# Solar Wind vs Earthquake Relative Event Rate Analysis  
### OMNI Daily Data • ISC-GEM Earthquake Catalog • Randomization Test (10 Seeds)

This project investigates whether solar wind conditions correlate with global earthquake occurrence.  
The analysis computes the **relative event rate (R)** across multiple solar-wind thresholds and compares the observed R-values to **10 randomized baselines** to determine whether deviations from 1 are statistically meaningful.

---

## 🔍 Research Goal

To test whether certain solar wind conditions (e.g., low density, high velocity, threshold crossings) correspond to higher or lower earthquake rates.

We compute:



\[
R = \frac{\text{Earthquake rate on condition days}}{\text{Earthquake rate on non-condition days}}
\]



Then we run **10 randomizations** of the OMNI dataset to determine whether deviations of R from 1 are:

- random noise  
- threshold artifacts  
- or potential physical signals  

---

## 📁 Data Sources

### **OMNI Daily Solar Wind Data**
- Magnetic Field  
- Density  
- Velocity  
- Cleaned by removing 999.9 / 9999.0 bad values  
- Date range: **1966–2021**

### **ISC-GEM Earthquake Catalog**
- Magnitude Mw ≥ 5.6  
- Matched to OMNI dates  
- Used to compute daily earthquake counts  

---

## 🧪 Method Overview

### 1. **Threshold Sweep**
For each variable (Density, Velocity, Magnetic Field), we compute thresholds from:



\[
V_{av-ad} \rightarrow 1.00
\]



For each threshold, we evaluate 6 conditions (C0–C5), such as:

- Days below threshold  
- Days above threshold  
- First day crossing threshold  
- Last day crossing threshold  
- Etc.  

### 2. **Relative Event Rate (R)**
For each condition:

- Count condition days (DC)  
- Count earthquakes on those days (EC)  
- Compute R  

### 3. **Randomization Test (10 Seeds)**
We shuffle OMNI values 10 times:

```python
df_randomized.sample(frac=1, random_state=seed)
