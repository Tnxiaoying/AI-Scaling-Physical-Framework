import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time
import sys

# --- 几何生成器 1: 谢尔宾斯基 (D=1.89) ---
def generate_sierpinski_carpet(L: int) -> (np.ndarray, int):
    # L 必须是 3 的幂
    k = int(np.log(L) / np.log(3) + 0.5) # +0.5 用于四舍五入
    base_size = 3
    seed = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)
    if k == 1: return seed, base_size
    carpet = seed
    for _ in range(1, k):
        carpet = np.kron(carpet, seed)
    return carpet, base_size

# --- 几何生成器 2: 维谢克 (D=1.46) ---
def generate_vicsek_fractal(L: int) -> (np.ndarray, int):
    k = int(np.log(L) / np.log(3) + 0.5)
    base_size = 3
    seed = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int8)
    if k == 1: return seed, base_size
    fractal = seed
    for _ in range(1, k):
        fractal = np.kron(fractal, seed)
    return fractal, base_size

# --- 几何生成器 3: 5x5 地毯 (D=1.97) ---
def generate_5x5_carpet(L: int) -> (np.ndarray, int):
    k = int(np.log(L) / np.log(5) + 0.5)
    base_size = 5
    seed = np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]], dtype=np.int8)
    if k == 1: return seed, base_size
    carpet = seed
    for _ in range(1, k):
        carpet = np.kron(carpet, seed)
    return carpet, base_size

# --- 几何生成器 4: 棋盘格 (D=2.0) ---
def generate_chessboard(L: int) -> (np.ndarray, int):
    grid = np.zeros((L, L), dtype=np.int8)
    grid[::2, ::2] = 1
    grid[1::2, 1::2] = 1
    base_size = 3 if L % 3 == 0 else 2
    return grid, base_size

# --- 几何生成器 5: 带边框的实心块 (D=2.0) ---
def generate_solid_with_border(L: int) -> (np.ndarray, int):
    grid = np.ones((L, L), dtype=np.int8)
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    base_size = 3 if L % 3 == 0 else 2
    return grid, base_size

# --- 测量与模拟工具 ---
def calculate_deff(grid: np.ndarray, base_size: int) -> float:
    L = grid.shape[0]
    factors = [i for i in range(2, L // 2 + 1) if L % i == 0]
    box_sizes = sorted(list(set(factors + [base_size**i for i in range(1, 6) if base_size**i < L])))
    if not box_sizes: return 2.0
    counts, sizes = [], []
    for s in box_sizes:
        if L % s != 0: continue
        N_s = 0
        for i in range(0, L, s):
            for j in range(0, L, s):
                if np.any(grid[i:i+s, j:j+s]):
                    N_s += 1
        if N_s > 0:
            counts.append(N_s)
            sizes.append(1.0 / s)
    if len(counts) < 2: return 2.0
    log_inv_s = np.log(sizes)
    log_counts = np.log(counts)
    try:
        slope, _ = np.polyfit(log_inv_s, log_counts, 1)
        return slope
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

def calculate_a_eff(grid: np.ndarray) -> int:
    cooling = (grid == 0)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbors_of_cooling = convolve2d(grid, kernel, mode='same', boundary='fill')
    a_eff = np.sum(neighbors_of_cooling[cooling])
    return int(a_eff)

def run_simulation(grid: np.ndarray, 
                   heatmap: np.ndarray, # <--- 修正：v4 应该使用 heatmap
                   max_iter: int = 50000, 
                   tolerance: float = 1e-6) -> (float, float, int):
    
    # --- 修正 v4/v5/v6 的 run_simulation ---
    # (确保我们使用的是最新的 v6/v7 版本)
    k_thermal = 0.1
    kernel = k_thermal * np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    temperature = np.zeros_like(grid, dtype=float)
    temp_old = np.zeros_like(grid, dtype=float)
    is_ctu = (grid == 1)
    is_cooling = (grid == 0)
    effective_heatmap = heatmap * is_ctu # 只在 CTU 上产热

    for i in range(max_iter):
        np.copyto(temp_old, temperature)
        
        # a. 产热
        temperature += effective_heatmap 
        
        # b. 散热
        diffusion = convolve2d(temperature, kernel, mode='same', boundary='wrap')
        temperature += diffusion
        
        # c. 冷却
        temperature[is_cooling] = 0.0
        
        # d. 检查稳态
        if i % 200 == 0: 
            change = np.max(np.abs(temperature - temp_old))
            if change < tolerance and i > 0:
                T_max = np.max(temperature)
                T_avg = np.mean(temperature[is_ctu]) 
                return T_max, T_avg, i
                
    T_max = np.max(temperature[is_ctu]) if np.any(is_ctu) else 0
    T_avg = np.mean(temperature[is_ctu]) if np.any(is_ctu) else 0
    print(f"    -> 警告: 已达到最大迭代次数 {max_iter}，模拟未收敛。")
    return T_max, T_avg, max_iter

# --- 主执行代码 (v4)：最终基准测试 (这是有 BUG 的绘图部分) ---
if __name__ == "__main__":
    
    # --- 1. 设置实验参数 ---
    geometries_to_test = [
        ("Vicsek",       generate_vicsek_fractal,    81,  np.log(5)/np.log(3)),
        ("Sierpinski",   generate_sierpinski_carpet, 81,  np.log(8)/np.log(3)),
        ("5x5 Carpet",   generate_5x5_carpet,        125, np.log(24)/np.log(5)),
        ("Chessboard",   generate_chessboard,        81,  2.0),
        ("Solid w/ Border", generate_solid_with_border, 81,  2.0)
    ]
    
    P_TOTAL = 1000.0 # v4/v5/v6 使用的是 P_TOTAL
    
    results = [] 

    print("--- Starting Final Benchmark Script (v4) ---")
    
    # --- 2. 循环执行所有实验 ---
    for (name, gen_func, L, theory_deff) in geometries_to_test:
        print(f"\n--- Processing: {name} (L={L}) ---")
        
        grid, base_size = gen_func(L)
        print(f"    Grid shape: {grid.shape}")
        
        V_eff = np.sum(grid)
        A_eff = calculate_a_eff(grid)
        D_eff = calculate_deff(grid, base_size=base_size)
        if np.isnan(D_eff): D_eff = theory_deff
            
        print(f"    V_eff (N_CTUs): {V_eff}")
        print(f"    A_eff (Boundary): {A_eff}")
        
        VA_ratio = V_eff / A_eff if A_eff > 0 else np.inf
        print(f"    V_eff / A_eff Ratio: {VA_ratio:.4f}")
        print(f"    D_eff (Measured): {D_eff:.4f} (Theory: {theory_deff:.4f})")
        
        # --- 模拟 (使用 v6/v7 的均匀产热) ---
        p_gen_coop = P_TOTAL / V_eff if V_eff > 0 else 0
        heatmap_coop = np.full(grid.shape, p_gen_coop)

        start_time = time.time()
        T_max, T_avg, iters = run_simulation(grid, heatmap_coop, max_iter=80000)
        
        if V_eff == 0 or np.isnan(T_avg):
            print("    -> 模拟失败 (V_eff=0 或 T_avg=nan)，跳过。")
            continue
            
        R_th = T_avg / P_TOTAL # (T_avg / P_eff)
        
        print(f"    Sim Time: {time.time() - start_time:.2f}s (Iters: {iters})")
        print(f"    T_avg (Global): {T_avg:.4f}")
        print(f"    R_th (T_avg/P_total): {R_th:.6f}")
        
        results.append({
            "name": name,
            "D_eff": D_eff,
            "R_th": R_th,
            "VA_ratio": VA_ratio
        })

    # ==================================================================
    # === 从这里开始，是 v4.1 的“修复补丁”绘图代码 ===
    # ==================================================================
    
    print("\n--- All experiments complete, generating final plots ---")

    # --- 3. 准备数据 ---
    if len(results) < 2:
        print("数据不足，无法绘图。")
        sys.exit()
        
    # --- 关键修复：将结果转换为 NumPy 数组以便排序 ---
    names = [r["name"] for r in results]
    d_eff_data = np.array([r["D_eff"] for r in results])
    r_th_data = np.array([r["R_th"] for r in results])
    va_ratio_data = np.array([r["VA_ratio"] for r in results])

    # --- 关键修复：按 X 轴 (VA_ratio) 排序！ ---
    sort_indices = np.argsort(va_ratio_data)
    
    # 使用排序后的索引来重新排列所有数据
    sorted_names = [names[i] for i in sort_indices]
    sorted_d_eff = d_eff_data[sort_indices]
    sorted_r_th = r_th_data[sort_indices]
    sorted_va_ratio = va_ratio_data[sort_indices]

    # --- 4. 绘制两张最终图表 ---
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Figure 2: The True Bottleneck in the $\delta$-Wind Tunnel', fontsize=20, y=0.97)

    # --- 图 1 (The "Real Law"): R_th vs. V_eff/A_eff ---
    # 现在我们使用 "sorted_" 变量来绘图
    ax1.plot(sorted_va_ratio, sorted_r_th, 'ro-', label='$R_{th}$ (Simulated)') # <-- 已修复
    ax1.set_xlabel('$V_{eff} / A_{eff}$ (Heat Source / Heat Sink Ratio)', fontsize=14)
    ax1.set_ylabel('Equivalent Thermal Resistance $R_{th}$ (Lower is Better)', fontsize=14)
    ax1.set_title('The "Real Law": $R_{th}$ vs. $V_{eff}/A_{eff}$', fontsize=18)
    ax1.grid(True, linestyle='--')
    
    # 拟合直线 (使用排序后的数据)
    z1 = np.polyfit(sorted_va_ratio, sorted_r_th, 1)
    p1 = np.poly1d(z1)
    r_corr = np.corrcoef(sorted_va_ratio, sorted_r_th)[0,1]
    ax1.plot(sorted_va_ratio, p1(sorted_va_ratio), "b--", label=f"Linear Fit ($R^2$={r_corr**2:.4f})")
    ax1.legend()

    # --- 图 2 (The "Deff Story"): R_th vs. D_eff ---
    # (使用原始数据，并改为散点图)
    ax2.plot(d_eff_data, r_th_data, 'go', markersize=10, label='$R_{th}$ (Simulated)') # <-- 改为散点图
    ax2.set_xlabel('$D_{eff}$ (Measured Effective Dimension)', fontsize=14)
    ax2.set_ylabel('$R_{th}$ (Lower is Better)', fontsize=14)
    ax2.set_title('The $D_{eff}$ Story (Not Monotonic)', fontsize=18)
    ax2.grid(True, linestyle='--')
    
    # 为每个点添加标签
    for i, name in enumerate(names):
        ax2.annotate(name, (d_eff_data[i], r_th_data[i]),
                     textcoords="offset points", xytext=(5, 5), ha='left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig("Figure_2_Delta_Bottleneck.png", dpi=300, bbox_inches='tight')
    print("Saved: Figure_2_Delta_Bottleneck.png")
    plt.show()
