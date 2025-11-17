import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata # <-- 新增
import time

# --- 1. 模拟器核心函数 (与之前相同) ---
def simulate_room(N, L, R, r, T=2000, T_warmup=200, step_size=1.0):
    # (此函数与 hamster_room+5-seed.py 相同，未修改)
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

# --- 2. 实验参数 (与之前相同) ---
N_seeds = 5
L_values = [8.0, 10.0, 14.0]
R_values = [0.1, 0.2]
N_values = [5, 10, 20, 40]
r_values = [0.3, 0.5, 0.7]
T = 2000
T_warmup = 200
STEP_SIZE = 1.0

start_time = time.time()
print(f"Starting 72 * {N_seeds} = 360 simulations...")

# --- 3. 执行模拟 (***已修改***) ---
# (修改了 'final_results' 中保存的变量)
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
        avg_rho = simulate_room(N, L, R, r, T, T_warmup, STEP_SIZE)
        rho_seeds.append(avg_rho)
    
    rho_mean = np.mean(rho_seeds)
    rho_std = np.std(rho_seeds)
    
    final_results.append({
        'L': L,
        'S': S,
        'R': R,
        'a': a,
        'N': N,
        'r': r,
        'Na': N * a,            # 新增: X轴 (密度)
        'one_over_S': 1.0 / S,  # 新增: Y轴 (压强)
        'rho_mean': rho_mean,   # Z轴 (拥挤度)
        'rho_std': rho_std
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Completed {i+1} / 72 parameter sets...")

end_time = time.time()
print(f"\nSimulations complete! Total time: {end_time - start_time:.2f} seconds")

# --- 4. 转换数据 (***已修改***) ---
# (修改了 'dtype' 以包含新变量)
results_dtype = [('L', 'f8'), ('S', 'f8'), ('R', 'f8'), ('a', 'f8'), 
                 ('N', 'i4'), ('r', 'f8'), ('Na', 'f8'), 
                 ('one_over_S', 'f8'), ('rho_mean', 'f8'), ('rho_std', 'f8')]
data = np.array([(d['L'], d['S'], d['R'], d['a'], d['N'], d['r'], 
                  d['Na'], d['one_over_S'], d['rho_mean'], d['rho_std']) 
                 for d in final_results], dtype=results_dtype)

# --- 5. 绘图 (***完全重写***) ---

plt.style.use('seaborn-v0_8-whitegrid')

# 我们必须选择一个 r ("温度") 切片来绘图
R_FILTER = 0.5 
plot_data = data[data['r'] == R_FILTER]
print(f"\nGenerating plots for r = {R_FILTER} ('temperature' slice)...")

# --- 图 1 (步骤三): "状态方程" 1D 切片 ---
# (Figure_hamster.png 的“正确”画法)
plt.figure(figsize=(10, 6))
colors = {8.0: 'blue', 10.0: 'green', 14.0: 'red'}

for L_val in L_values:
    S_val = L_val**2
    # 筛选出这个 'S' (压强) 和 'r' (温度) 的所有数据点
    subset = plot_data[plot_data['L'] == L_val]
    # 按 X 轴 (N*a) 排序
    subset.sort(order='Na')
    
    # 绘制曲线
    plt.plot(subset['Na'], subset['rho_mean'], 
             label=f'S={S_val:.0f} (Low Pressure)' if L_val==14.0 else f'S={S_val:.0f}', 
             color=colors[L_val], marker='o', linestyle='-')
    
    # 绘制误差条
    plt.fill_between(subset['Na'], 
                     subset['rho_mean'] - subset['rho_std'], 
                     subset['rho_mean'] + subset['rho_std'], 
                     color=colors[L_val], alpha=0.2)

plt.title(f'Figure 3: "Equation of State" Slices (at r={R_FILTER})', fontsize=14)
plt.xlabel('Density (N*a)', fontsize=12)
plt.ylabel('Crowding (ρ_mean)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_3_EOS_Slices.png", dpi=300, bbox_inches='tight')
print("Saved: Figure_3_EOS_Slices.png")
plt.show()

# --- 图 2 (步骤二): "相图" 2D 等高线图 ---
plt.figure(figsize=(10, 7))

# 提取 X, Y, Z 轴数据
x = plot_data['Na']          # X = 密度
y = plot_data['one_over_S']  # Y = 压强
z = plot_data['rho_mean']    # Z = 拥挤度 (颜色)

# 创建插值网格
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X_grid, Y_grid = np.meshgrid(xi, yi)

# 插值
Z_grid = griddata((x, y), z, (X_grid, Y_grid), method='cubic')

# 绘制等高线图
plt.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap='viridis')
plt.colorbar(label='Crowding (ρ_mean)')

# 绘制原始数据点（可选，用于检查）
# plt.scatter(x, y, c='red', s=5, label='Data Points')

plt.title(f'Figure B1: Crowding Phase Diagram (at r={R_FILTER})', fontsize=14)
plt.xlabel('Density (N*a)', fontsize=12)
plt.ylabel('Pressure (1/S)', fontsize=12)
plt.tight_layout()
plt.savefig("Figure_B1_Crowding_Phase.png", dpi=300, bbox_inches='tight')
print("Saved: Figure_B1_Crowding_Phase.png")
plt.show()

print("\nPhase diagram plots generated.")
