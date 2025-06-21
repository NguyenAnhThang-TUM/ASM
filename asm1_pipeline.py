# ASM1 Full Implementation with Mock Sensor Data Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------
# ASM1 MODEL IMPLEMENTATION
# ------------------------

def asm1_odes(state, t, p):
    # State variables: SS, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK
    SS, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK = state

    # Parameters (subset for clarity)
    mu_H = p['mu_H']
    K_S = p['K_S']
    b_H = p['b_H']
    Y_H = p['Y_H']
    k_dec = p['b_A']

    # Process rates (simplified subset)
    rho1 = mu_H * SS / (K_S + SS) * SO / (0.2 + SO) * XBH              # Aerobic growth of heterotrophs
    rho2 = b_H * XBH                                                  # Decay of heterotrophs
    rho3 = k_dec * XBA                                               # Decay of autotrophs

    # Differential equations
    dSS = -rho1
    dXS = 0
    dXBH = rho1 - rho2
    dXBA = -rho3
    dXP = 0.08 * rho2 + 0.08 * rho3
    dSO = -1.42 * rho1
    dSNO = 0
    dSNH = -0.1 * rho1
    dSND = 0
    dXND = 0
    dSALK = 0.5 * rho1

    return [dSS, dXS, dXBH, dXBA, dXP, dSO, dSNO, dSNH, dSND, dXND, dSALK]

# Initial state (mg/L)
state0 = [30, 60, 20, 5, 0, 8, 2, 25, 6, 3, 7]

# Parameters
params = {
    'mu_H': 3.2,
    'K_S': 20,
    'b_H': 0.62,
    'Y_H': 0.67,
    'b_A': 0.2
}

# Time (days)
t = np.linspace(0, 10, 200)

# Solve ODE
solution = odeint(asm1_odes, state0, t, args=(params,))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t, solution[:, 0], label='SS')
plt.plot(t, solution[:, 2], label='XBH')
plt.plot(t, solution[:, 4], label='XP')
plt.plot(t, solution[:, 5], label='SO')
plt.xlabel("Time (days)")
plt.ylabel("Concentration (mg/L)")
plt.legend()
plt.grid()
plt.title("ASM1 Simulation")
plt.show()

# ------------------------
# MOCK SENSOR DATA PIPELINE
# ------------------------

# Create mock WWTP sensor dataset
dates = pd.date_range(start="2024-01-01", periods=500, freq="H")
data = pd.DataFrame({
    "timestamp": dates,
    "Flow_m3h": np.random.normal(1200, 150, size=500),
    "DO_mgL": np.clip(np.random.normal(2.5, 0.5, size=500), 0, None),
    "NH4_mgL": np.random.normal(20, 3, size=500),
    "NO3_mgL": np.random.normal(12, 2, size=500),
    "COD_mgL": np.random.normal(400, 50, size=500),
    "Temp_C": np.random.normal(17, 3, size=500)
})

# Basic plotting
data.set_index("timestamp")[["DO_mgL", "NH4_mgL", "NO3_mgL"]].plot(figsize=(14, 6))
plt.title("Sensor Trends: DO, NH4, NO3")
plt.grid()
plt.show()

# Correlation heatmap
sns.heatmap(data.drop("timestamp", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# PCA analysis
features = data.drop("timestamp", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.3)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of WWTP Sensor Data")
plt.grid()
plt.show()
