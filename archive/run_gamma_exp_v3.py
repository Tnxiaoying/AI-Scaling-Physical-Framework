import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time
import sys
import math

# --- 几何生成器 (Solid w/ Border L=81) ---
def generate_solid_with_border(L: int) -> np.ndarray:
    """
    生成一个 L x L 的实心块，但在最外层有一圈 "冷却" 边框。
    (1=CTU, 0=冷却)
    """
    grid = np.ones((L, L), dtype=np.int8)
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid

# --- 散热模拟器 (与 v2 脚本相同) ---
def run_simulation(grid: np.ndarray, 
                   heatmap: np.ndarray, 
                   max_iter: int = 50000, 
                   tolerance: float = 1e-6) -> (float, float, int):
    """
    运行热稳态模拟。
    (T_max, T_avg, 迭代次数)
    """
    k_thermal = 0.1
    kernel = k_thermal * np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    temperature = np.zeros_like(grid, dtype=float)
    temp_old = np.zeros_like(grid, dtype=float)
    is_ctu = (grid == 1)
    is_cooling = (grid == 0)
    
    effective_heatmap = heatmap * is_ctu

    for i in range(max_iter):
        np.copyto(temp_old, temperature)
        
        temperature += effective_heatmap
        diffusion = convolve2d(temperature, kernel, mode='same', boundary='wrap')
        temperature += diffusion
        temperature[is_cooling] = 0.0
        
        if i % 200 == 0: 
            change = np.max(np.abs(temperature - temp_old))
            if change < tolerance and i > 0:
                T_max = np.max(temperature)
                # [修正] T_avg 应该只计算 CTU 区域
                T_avg_ct = np.mean(temperature[is_ctu]) if np.any(is_ctu) else 0
                return T_max, T_avg_ct, i
                
    T_max = np.max(temperature[is_ctu]) if np.any(is_ctu) else 0
    T_avg_ct = np.mean(temperature[is_ctu]) if np.any(is_ctu) else 0
    return T_max, T_avg_ct, max_iter

# --- 主执行代码 (v3)：Gamma (γ) 曲线扫描 ---
if __name__ == "__main__":
    
    # --- 1. 设置实验参数 ---
    L = 81 # 硬件: Solid w/ Border L=81
    P_TOTAL = 1000.0 # 总功耗
    
    print("--- Starting Gamma (γ) Curve Experiment (v3) ---")
    
    # --- 2. 生成硬件 (δ 固定为 "差") ---
    print(f"Generating hardware: Solid w/ Border L={L}")
    grid = generate_solid_with_border(L)
    N_CTUs_Total = np.sum(grid) # 79*79 = 6241
    print(f"    Grid shape: {grid.shape}, Total CTUs (V_eff): {N_CTUs_Total}")
    
    # --- 3. 定义 f 扫描点 ---
    # (热点边长, 近似f值)
    hotspot_sizes = [79, 56, 40, 28, 20]
    approx_f_labels = ["1", "1/2", "1/4", "1/8", "1/16"]
    
    results = []
    baseline_R_th = 0.0
    baseline_T_max = 0.0

    # --- 4. 循环执行实验 ---
    for i, hotspot_size in enumerate(hotspot_sizes):
        f_label = approx_f_labels[i]
        print(f"\n--- Running: Conflict Mode (f ≈ {f_label}) ---")
        
        # a. 创建 "中心热点" 产热图
        heatmap_conflict = np.zeros(grid.shape)
        
        center_idx = L // 2 # 81 // 2 = 40
        half_size = hotspot_size // 2 
        
        # 修正奇偶数尺寸
        start_idx = center_idx - (hotspot_size // 2)
        end_idx = start_idx + hotspot_size
        
        # 确保热点索引在 CTU 区域内 (1 到 L-1)
        start_idx_clamped = max(1, start_idx)
        end_idx_clamped = min(L-1, end_idx)
        
        # 找出这个热点区域内的 CTU 数量
        hotspot_grid_slice = grid[start_idx_clamped:end_idx_clamped, start_idx_clamped:end_idx_clamped]
        N_hotspot_CTUs = np.sum(hotspot_grid_slice)
        
        if N_hotspot_CTUs == 0:
            print(f"Error: Hotspot (f={f_label}) contains no CTUs. Skipping.")
            continue
            
        f_actual = N_hotspot_CTUs / N_CTUs_Total
        
        # 计算热点产热率
        p_gen_hotspot = P_TOTAL / N_hotspot_CTUs
        
        # 将产热率 "注入" 到热图中
        heatmap_conflict[start_idx_clamped:end_idx_clamped, start_idx_clamped:end_idx_clamped] = p_gen_hotspot
        
        print(f"    Hotspot: Center {hotspot_size}x{hotspot_size} block")
        print(f"    Hotspot CTUs: {N_hotspot_CTUs} (f_actual = {f_actual:.4f})")
        print(f"    P_gen (Hotspot): {p_gen_hotspot:.6f}")

        # b. 运行模拟 (f=1 可能需要更多迭代)
        iters_needed = 50000 if hotspot_size < 79 else 100000 
        start_time = time.time()
        T_max, T_avg, iters = run_simulation(grid, heatmap_conflict, max_iter=iters_needed)

        # c. 计算热阻
        R_th = T_avg / P_TOTAL
        
        print(f"    Sim Time: {time.time() - start_time:.2f}s (Iters: {iters})")
        print(f"    T_avg: {T_avg:.4f}")
        print(f"    T_max: {T_max:.4f}")
        print(f"    R_th: {R_th:.6f}")
        
        # 存储结果
        result_data = {
            "f_label": f_label,
            "f_actual": f_actual,
            "R_th": R_th,
            "T_avg": T_avg,
            "T_max": T_max
        }
        results.append(result_data)
        
        # 存储基准值 (f=1)
        if f_label == "1":
            baseline_R_th = R_th
            baseline_T_max = T_max

    # --- 5. 打印最终结果表格 ---
    print("\n--- Final Results (v3): Gamma(f) Curve on 'Solid' Hardware ---")
    print("==================================================================================")
    print(f"Baseline (f=1.0): R_th = {baseline_R_th:.6f}, T_max = {baseline_T_max:.4f}")
    print("----------------------------------------------------------------------------------")
    print("f (approx) | f (actual) | R_th (avg) | F_avg (Penalty) | gamma (1/F) | F_max (Penalty)")
    print("----------------------------------------------------------------------------------")
    
    # --- [新增] 用于绘图的数据列表 ---
    plot_f_actual = []
    plot_gamma = []
    plot_f_max = []
    # --------------------------------

    if baseline_R_th == 0:
        print("Error: Baseline (f=1) run failed or produced R_th=0.")
    else:
        for res in results:
            f_label = res['f_label']
            f_actual = res['f_actual']
            R_th = res['R_th']
            T_max = res['T_max']
            
            F_avg = R_th / baseline_R_th
            gamma = 1.0 / F_avg
            F_max = T_max / baseline_T_max
            
            print(f"f ≈ {f_label:<7} | {f_actual:<10.4f} | {R_th:<10.6f} | {F_avg:<15.2f}x | {gamma:<11.4f} | {F_max:<15.2f}x")
            
            # --- [新增] 存储数据用于绘图 ---
            plot_f_actual.append(f_actual)
            plot_gamma.append(gamma)
            plot_f_max.append(F_max)
            # --------------------------------
            
    print("==================================================================================")

    # --- [新增] 绘制 论文图 3 ---
    if baseline_R_th > 0 and len(plot_f_actual) > 1:
        print("\nGenerating Figure 4: Physical Gamma (γ_phys) Curve...")
        
        # 翻转数据，使 f 从 0.06 -> 1.0 (X轴递增)
        plot_f_actual.reverse()
        plot_gamma.reverse()
        plot_f_max.reverse()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Figure 4: $\gamma_{phys}$ (Software Efficiency) vs. Load Concentration ($f$)', fontsize=18, y=0.98)

        # --- 子图 1: Gamma (γ) 崩塌曲线 (论文核心图) ---
        ax1.plot(plot_f_actual, plot_gamma, 'bo-', label='$\gamma_{phys} = 1 / F_{avg}$')
        ax1.set_xlabel('Load Concentration Factor, $f$ (Fraction of CTUs activated)', fontsize=12)
        ax1.set_ylabel('Software Efficiency Factor, $\gamma_{phys}$', fontsize=12)
        ax1.set_title('a) Efficiency Collapse ($\gamma_{phys}$ vs. $f$)', fontsize=15)
        ax1.grid(True, linestyle='--')
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)
        ax1.legend()

        # --- 子图 2: 峰值温度惩罚 (F_max) ---
        ax2.plot(plot_f_actual, plot_f_max, 'ro-', label='$F_{max} = T_{max}(f) / T_{max}(1)$')
        ax2.set_xlabel('Load Concentration Factor, $f$', fontsize=12)
        ax2.set_ylabel('Peak Temperature Penalty, $F_{max}$ (Higher is Worse)', fontsize=12)
        ax2.set_title('b) Peak Temperature Penalty ($F_{max}$ vs. $f$)', fontsize=15)
        ax2.grid(True, linestyle='--')
        ax2.set_xlim(0, 1.05)
        ax2.legend()
        
        # --- 保存并显示 ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        output_filename = "Figure_4_Gamma_Phys.png"
        plt.savefig(output_filename)
        print(f"Successfully saved image: {output_filename}")
        
        plt.show()
    else:
        print("\nSkipping plot generation due to missing baseline data or insufficient data points.")
    # ---------------------------------
