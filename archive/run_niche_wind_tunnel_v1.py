import numpy as np
import time
import matplotlib.pyplot as plt # <--- 导入绘图库
import sys

def run_niche_simulation(R: int, M: int, n_trials: int = 10000) -> dict:
    """
    Runs the "Niche Wind Tunnel" simulation.
    
    Parameters:
    R (int): Total number of resource dimensions (e.g., 100 channels)
    M (int): Total number of tasks (e.g., 50 "ants")
    n_trials (int): Number of trials to run for statistical averaging
    
    Returns:
    A dictionary containing averaged metrics.
    """
    
    # Accumulators for results over all trials
    total_gamma_shared = 0.0
    total_gamma_exclusive = 0.0
    total_f_niche = 0.0

    for _ in range(n_trials):
        # 1. Task Routing: M tasks randomly choose R resources
        #    (Collisions are allowed)
        choices = np.random.randint(0, R, size=M)
        
        # 2. Count Collisions
        #    (Use bincount for a fast histogram of choices on R)
        resource_usage = np.bincount(choices, minlength=R)
        
        # --- 3. Calculate Metrics (using v2 definitions) ---
        
        # Task i chose resource r, which has k_r tasks on it.
        # Task i's "shared bandwidth" output = 1 / k_r
        # Task i's "exclusive" output = 1 (if k_r=1) or 0 (if k_r > 1)
        
        E_total_shared = 0.0
        E_total_exclusive = 0.0
        
        # Iterate over the M task choices
        for task_choice in choices:
            k = resource_usage[task_choice] # Collision count for this task's resource
            E_total_shared += (1.0 / k)
            if k == 1:
                E_total_exclusive += 1.0
        
        # E_max (ideal output) = M
        gamma_shared = E_total_shared / M
        gamma_exclusive = E_total_exclusive / M

        # f_niche (Resource Usage)
        R_active = np.count_nonzero(resource_usage)
        f_niche = R_active / R
        
        # Accumulate
        total_gamma_shared += gamma_shared
        total_gamma_exclusive += gamma_exclusive
        total_f_niche += f_niche
        
    # --- 4. Return the average over n_trials ---
    return {
        "gamma_shared": total_gamma_shared / n_trials,
        "gamma_exclusive": total_gamma_exclusive / n_trials,
        "f_niche": total_f_niche / n_trials,
    }

# --- 主执行代码 (v1)：Niche Wind Tunnel ---
if __name__ == "__main__":
    
    # --- [已修复] 重新添加了被遗漏的参数 ---
    R_RESOURCES = 100  # Fixed number of resource dimensions
    N_TRIALS = 10000   # Number of statistical trials per M
    
    # Scan "Task Pressure" (M) from M/R = 0.01 to 2.0
    M_sweep = list(range(1, 11)) + \
              list(range(15, 51, 5)) + \
              list(range(60, 101, 10)) + \
              list(range(120, 201, 20))
    # ---------------------------------------------
              
    print("--- Starting Niche (γ_niche) Wind Tunnel (v1) ---")
    print(f"Hardware: R_RESOURCES = {R_RESOURCES}")
    print(f"Method: Sweeping M (Tasks) from 1 to {M_sweep[-1]}")
    print(f"Stats: N_TRIALS = {N_TRIALS} per M")
    
    print("\n" + "="*70)
    print("M (Tasks) | M/R Ratio | f_niche (Usage) | gamma_shared | gamma_exclusive")
    print("-"*70)
    
    start_time = time.time()
    
    # --- [新增] 用于绘图的数据列表 ---
    plot_M_R_Ratio = []
    plot_gamma_shared = []
    plot_gamma_exclusive = []
    plot_f_niche = []
    # --------------------------------
    
    for M_TASKS in M_sweep:
        
        # 运行模拟
        avg_results = run_niche_simulation(R_RESOURCES, M_TASKS, N_TRIALS)
        
        M_R_Ratio = M_TASKS / R_RESOURCES
        f_niche = avg_results['f_niche']
        g_shared = avg_results['gamma_shared']
        g_excl = avg_results['gamma_exclusive']
        
        # 存储并打印
        print(f"{M_TASKS:<9} | {M_R_Ratio:<9.2f} | {f_niche:<15.4f} | {g_shared:<12.4f} | {g_excl:<15.4f}")

        # --- [新增] 存储数据用于绘图 ---
        plot_M_R_Ratio.append(M_R_Ratio)
        plot_gamma_shared.append(g_shared)
        plot_gamma_exclusive.append(g_excl)
        plot_f_niche.append(f_niche)
        # --------------------------------

    print("="*70)
    print(f"\nTotal Simulation Time: {time.time() - start_time:.2f}s")
    
    # --- [新增] 绘制 论文图 4 ---
    if len(plot_M_R_Ratio) > 1:
        print("\nGenerating Figure A1: Niche Gamma (γ_niche) Curve...")
        
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
        # --- [已修复] 添加 r'' 来处理 $\gamma$ ---
        fig.suptitle(r'Figure A1: $\gamma_{niche}$ (Abstract Efficiency) vs. Task Pressure ($M/R$)', fontsize=18, y=0.98) 

        # --- 绘制核心图：Gamma (共享) vs M/R ---
        ax1.plot(plot_M_R_Ratio, plot_gamma_shared, 'bo-', label=r'$\gamma_{niche}$ (Shared Bandwidth Model)')
        ax1.set_xlabel('Task Pressure Ratio, $M/R$ (Tasks / Resources)', fontsize=12)
        ax1.set_ylabel(r'Abstract Efficiency Factor, $\gamma_{niche}$', fontsize=12)
        ax1.set_title(r'Efficiency Collapse ($\gamma_{niche}$ vs. $M/R$)', fontsize=15)
        
        # --- [可选] 绘制 Gamma (排他) 作为对比 ---
        ax1.plot(plot_M_R_Ratio, plot_gamma_exclusive, 'go--', label=r'$\gamma_{niche}$ (Exclusive Model)', alpha=0.7)
        
        # --- [可选] 绘制 f_niche (资源使用率) ---
        # ax1.plot(plot_M_R_Ratio, plot_f_niche, 'rs--', label=r'$f_{niche}$ (Resource Usage)', alpha=0.5)

        ax1.grid(True, linestyle='--')
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        
        # --- 保存并显示 ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) 
        
        output_filename = "Figure_A1_Gamma_Niche.png"
        plt.savefig(output_filename)
        print(f"Successfully saved image: {output_filename}")
        
        plt.show()
    else:
        print("\nSkipping plot generation due to insufficient data points.")
    # ---------------------------------
