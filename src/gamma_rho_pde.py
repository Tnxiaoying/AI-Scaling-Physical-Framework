import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import time

# --- 1. 2D Hotspot Simulator (PDE Solver) ---

def run_hotspot_sim(f, N=50, Q_total=100.0, kappa=1.0, max_iter=10000, tol=1e-6):
    """
    Runs 2D hotspot simulation (solves ∇²T = -q/κ).
    
    f: Load concentration (1.0 = uniform, < 1.0 = hotspot)
    N: Grid size (N x N)
    Q_total: Total heat source power
    
    Returns:
    T_avg: Average temperature at steady state
    """
    
    # 1. Initialize grids
    T = np.zeros((N, N))
    q = np.zeros((N, N))
    h = 1.0 / (N - 1) # grid step (assuming L=1.0)
    
    # 2. Construct heat source q(x, y; f)
    if f == 1.0:
        # Uniform load
        q[:, :] = Q_total / (N * N)
    else:
        # Concentrated load (hotspot)
        hotspot_area_target = f * (N * N)
        size = int(np.sqrt(hotspot_area_target))
        if size == 0: size = 1
        
        # Center it
        start = (N - size) // 2
        end = start + size
        
        # Conserve total heat Q_total
        actual_hotspot_area = size * size
        if actual_hotspot_area > 0:
            q[start:end, start:end] = Q_total / actual_hotspot_area
        else:
            q[N//2, N//2] = Q_total # Failsafe

    # 3. Iterative solve for steady state (Jacobi/Gauss-Seidel)
    # Boundary conditions: T=0 on all 4 borders (cold plate)
    T_old = T.copy()
    for i in range(max_iter):
        T_new = T_old.copy()
        
        # Core PDE update
        T_new[1:-1, 1:-1] = 0.25 * (T_old[2:, 1:-1] + T_old[:-2, 1:-1] + 
                                   T_old[1:-1, 2:] + T_old[1:-1, :-2] + 
                                   q[1:-1, 1:-1] * (h**2) / kappa)
        
        # Check convergence
        max_diff = np.max(np.abs(T_new - T_old))
        if max_diff < tol:
            break
        
        T_old = T_new
        
    # 4. Return average temperature
    return np.mean(T_new)

# --- 2. Run Simulation & Process Data ---

f_values = np.array([1.0, 0.5, 0.25, 0.125])
T_avg_values = []

print("--- Running 2D Hotspot PDE Simulation ---")
start_time = time.time()

for f in f_values:
    print(f"  Simulating f = {f}...")
    t_avg = run_hotspot_sim(f)
    T_avg_values.append(t_avg)
    
print(f"Simulation complete! Time: {time.time() - start_time:.2f} seconds\n")

# 3. Calculate F_avg and γ_phys
T_avg_values = np.array(T_avg_values)
T_avg_baseline = T_avg_values[0] # T_avg(f=1.0)
F_avg_values = T_avg_values / T_avg_baseline
gamma_values = 1.0 / F_avg_values

# 4. Map f to ρ_hat (normalized to [0, 1])
f_min = np.min(f_values)
rho_raw = 1.0 / f_values
# Normalization: (x - min) / (max - min)
rho_hat = (rho_raw - 1.0) / ((1.0 / f_min) - 1.0)

X_data = rho_hat
Y_data = gamma_values

# --- Print Data Table (for paper) ---
print("--- 'Real' Hotspot Experiment Data Table ---")
print(f"{'f':<8} | {'T_avg (proxy)':<15} | {'F_avg (Penalty)':<15} | {'γ_phys (Efficiency)':<15} | {'ρ_hat':<10}")
print("-" * 67)
for i in range(len(f_values)):
    print(f"{f_values[i]:<8.4f} | {T_avg_values[i]:<15.5f} | {F_avg_values[i]:<15.5f} | {gamma_values[i]:<15.5f} | {rho_hat[i]:<10.4f}")
print("-" * 67)


# --- 5. Fit γ(ρ) ---

# Define candidate models
def model_A(rho, k): return 1.0 / (1.0 + k * rho)
def model_B(rho, k): return np.exp(-k * rho)
def model_C(rho, k): return 1.0 - k * rho

models = {
    'A: 1 / (1 + kρ)': model_A,
    'B: exp(-kρ)': model_B,
    'C: 1 - kρ (linear)': model_C
}

fit_results = {}
print("\n--- γ(ρ) Fit Results (based on 'real' data) ---")

for name, model_func in models.items():
    popt, pcov = curve_fit(model_func, X_data, Y_data, p0=[0.5])
    k_fit = popt[0]
    y_pred = model_func(X_data, k_fit)
    r_squared = r2_score(Y_data, y_pred)
    
    fit_results[name] = {'k': k_fit, 'r2': r_squared, 'popt': popt}
    print(f"Model {name}:")
    print(f"  > Fit parameter k = {k_fit:.4f}")
    print(f"  > R² Score = {r_squared:.4f}")

print("-" * 67)

# --- 6. Visualization ---
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot "real" data points
plt.scatter(X_data, Y_data, label="'Real' Data (from PDE Sim)", color='red', s=100, zorder=5)

# Plot fit curves
rho_smooth = np.linspace(0, 1, 100)
colors = {'A': 'blue', 'B': 'green', 'C': 'gray'}
linestyles = {'A': '-', 'B': '--', 'C': ':'}

for name, result in fit_results.items():
    if result:
        model_name_short = name.split(':')[0]
        label = f"{name} (R²={result['r2']:.3f})"
        plt.plot(rho_smooth, models[name](rho_smooth, *result['popt']),
                 label=label,
                 color=colors[model_name_short],
                 linestyle=linestyles[model_name_short],
                 linewidth=2.5)

plt.title("Figure 5: γ(ρ) Function Fit (based on 'real' PDE data)", fontsize=14)
plt.xlabel("ρ_hat (Normalized Crowding)", fontsize=12)
plt.ylabel("γ_phys (Normalized Efficiency)", fontsize=12)
plt.legend(fontsize=11)
plt.ylim(min(Y_data) - 0.1, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_5_Gamma_Rho.png", dpi=300, bbox_inches='tight')
print("Saved: Figure_5_Gamma_Rho.png")
plt.show()
