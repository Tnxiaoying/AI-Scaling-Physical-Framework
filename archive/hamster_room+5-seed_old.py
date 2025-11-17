import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import time

# --- 1. Simulator Core Function ---
def simulate_room(N, L, R, r, T=2000, T_warmup=200, step_size=1.0):
    # Each call will use a new random state
    positions = np.random.rand(N, 2) * L
    rho_history = []
    max_pairs = N * (N - 1) / 2
    if max_pairs == 0:
        return 0.0

    for t in range(T):
        movers = np.random.rand(N) < r
        num_movers = np.sum(movers)
        
        if num_movers > 0:
            angles = np.random.rand(num_movers) * 2 * np.pi
            delta_x = np.cos(angles) * step_size
            delta_y = np.sin(angles) * step_size
            positions[movers, 0] += delta_x
            positions[movers, 1] += delta_y
            positions = np.clip(positions, 0, L)

        dist_matrix = squareform(pdist(positions))
        collisions_matrix = (dist_matrix < 2 * R) & (dist_matrix > 0)
        collisions_t = np.sum(collisions_matrix) / 2
        rho_t = collisions_t / max_pairs
        
        if t >= T_warmup:
            rho_history.append(rho_t)
            
    return np.mean(rho_history)

# --- 2. Power Law Fit Function ---
def power_law(x, c, beta):
    return c * (x**beta)

# --- 3. Experiment Parameters (N_seeds=5) ---
N_seeds = 5
L_values = [8.0, 10.0, 14.0]
R_values = [0.1, 0.2]
N_values = [5, 10, 20, 40]
r_values = [0.3, 0.5, 0.7]

T = 2000
T_warmup = 200
STEP_SIZE = 1.0

results = []
start_time = time.time()

print(f"Starting 72 * {N_seeds} = 360 simulations...")

# --- 4. Run Simulation (with seed loop) ---
param_combos = []
for L in L_values:
    for R in R_values:
        for N in N_values:
            for r in r_values:
                param_combos.append({'L': L, 'R': R, 'N': N, 'r': r})

final_results = []
for i, params in enumerate(param_combos):
    L, R, N, r = params['L'], params['R'], params['N'], params['r']
    S = L**2
    a = np.pi * R**2
    
    rho_seeds = []
    for seed in range(N_seeds):
        # Each call uses a new np.random state
        avg_rho = simulate_room(N, L, R, r, T, T_warmup, STEP_SIZE)
        rho_seeds.append(avg_rho)
    
    # Calculate mean and std dev
    rho_mean = np.mean(rho_seeds)
    rho_std = np.std(rho_seeds)
    
    final_results.append({
        'L': L,
        'S': S,
        'X_corrected': (N * a * r) / S, # X' = N*a*r / S
        'rho_mean': rho_mean,
        'rho_std': rho_std
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Completed {i+1} / 72 parameter sets...")

end_time = time.time()
print(f"\nSimulations complete! Total time: {end_time - start_time:.2f} seconds")

# Convert to structured numpy array
results_dtype = [('L', 'f8'), ('S', 'f8'), ('X_corrected', 'f8'), 
                 ('rho_mean', 'f8'), ('rho_std', 'f8')]
data = np.array([(d['L'], d['S'], d['X_corrected'], d['rho_mean'], d['rho_std']) 
                 for d in final_results], dtype=results_dtype)

# --- 5. Plotting (with error bars) ---
plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(12, 7)) # Figure size large enough for error bars

X_data_full = data['X_corrected']
Y_data_full = data['rho_mean']
Y_error = data['rho_std']

# Fit all data (using mean)
popt_full, pcov_full = curve_fit(power_law, X_data_full, Y_data_full, p0=[1.0, 1.0])
c_full, beta_full = popt_full

print("\n--- Global Fit Results (based on ρ_mean) ---")
print(f"Fit function: ρ_mean ≈ c * (N*a*r / S)^β")
print(f"Fit parameters: c = {c_full:.4f}, β = {beta_full:.4f}")

# Plot scatter + error bars
colors = {8.0: 'blue', 10.0: 'green', 14.0: 'red'}
for L_val in L_values:
    subset = data[data['L'] == L_val]
    plt.errorbar(subset['X_corrected'], subset['rho_mean'], yerr=subset['rho_std'],
                 fmt='o', # 'o' = marker
                 color=colors[L_val],
                 alpha=0.6,
                 capsize=5, # Error bar cap size
                 label=f'L={L_val} (S={L_val**2:.0f}) (mean ± 1 std)')

# Plot fit line
x_fit = np.linspace(min(X_data_full) * 0.9, max(X_data_full) * 1.1, 100)
y_fit = power_law(x_fit, c_full, beta_full)
plt.plot(x_fit, y_fit, 'k--', 
         label=f'Global Fit (β={beta_full:.3f})',
         zorder=10) # Ensure line is on top

plt.title('Figure B-v2: ρ vs (N·a·r / S) (with 5-seed error bars)', fontsize=14)
plt.xlabel("Corrected Crowding (X' = N·a·r / S)", fontsize=12)
plt.ylabel('Average Crowding (ρ_mean ± 1 std)', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("Figure_B2_Crowding_vs_Xprime.png", dpi=300, bbox_inches='tight')
print("Saved: Figure_B2_Crowding_vs_Xprime.png")
plt.show()
