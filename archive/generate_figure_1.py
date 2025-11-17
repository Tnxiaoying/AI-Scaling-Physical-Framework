import numpy as np
import matplotlib.pyplot as plt
import sys

# --- 几何生成器 (从 run_experiments.py 复制而来) ---

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

def generate_vicsek_fractal(L: int) -> (np.ndarray, int):
    k = int(np.log(L) / np.log(3) + 0.5)
    base_size = 3
    seed = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int8)
    if k == 1: return seed, base_size
    fractal = seed
    for _ in range(1, k):
        fractal = np.kron(fractal, seed)
    return fractal, base_size

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

def generate_chessboard(L: int) -> (np.ndarray, int):
    grid = np.zeros((L, L), dtype=np.int8)
    grid[::2, ::2] = 1
    grid[1::2, 1::2] = 1
    base_size = 3 if L % 3 == 0 else 2
    return grid, base_size

def generate_solid_with_border(L: int) -> (np.ndarray, int):
    grid = np.ones((L, L), dtype=np.int8)
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1, ] = 0
    base_size = 3 if L % 3 == 0 else 2
    return grid, base_size

# --- 主执行代码：生成并绘制图 1 ---
if __name__ == "__main__":
    
    # --- [移除] 中文字体设置 (不再需要) ---
    print("Generating Figure 1: Geometry Schematics...")
    
    # --- 1. 设置画布 ---
    # 我们使用 2x3 的网格来展示 5 个几何
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # --- [修复] 调整 y 值，防止标题被裁剪 ---
    fig.suptitle('Figure 1: The Five Core Geometries for the $\delta$-Wind Tunnel', fontsize=20, y=0.98) # <--- 从 1.02 修改为 0.98
    
    # 定义 cmap (1=CTU=黑色, 0=Cooling=白色)
    cmap = 'binary'

    # --- 2. 绘制 5 个子图 ---
    
    # (0, 0) Vicsek
    L_vicsek = 27 # k=3
    grid_vicsek, _ = generate_vicsek_fractal(L_vicsek)
    axs[0, 0].imshow(grid_vicsek, cmap=cmap)
    axs[0, 0].set_title(f"a) Vicsek (L={L_vicsek})\n$D_{{theory}} = 1.465$", fontsize=14)
    axs[0, 0].axis('off')
    
    # (0, 1) Sierpinski
    L_sierpinski = 27 # k=3
    grid_sierpinski, _ = generate_sierpinski_carpet(L_sierpinski)
    axs[0, 1].imshow(grid_sierpinski, cmap=cmap)
    axs[0, 1].set_title(f"b) Sierpinski (L={L_sierpinski})\n$D_{{theory}} = 1.893$", fontsize=14)
    axs[0, 1].axis('off')
    
    # (0, 2) 5x5 Carpet
    L_5x5 = 25 # k=2
    grid_5x5, _ = generate_5x5_carpet(L_5x5)
    axs[0, 2].imshow(grid_5x5, cmap=cmap)
    axs[0, 2].set_title(f"c) 5x5 Carpet (L={L_5x5})\n$D_{{theory}} = 1.975$", fontsize=14)
    axs[0, 2].axis('off')
    
    # (1, 0) Chessboard (好几何)
    L_chess = 27
    grid_chess, _ = generate_chessboard(L_chess)
    axs[1, 0].imshow(grid_chess, cmap=cmap)
    axs[1, 0].set_title(f"d) Chessboard (L={L_chess})\n$D_{{theory}} = 2.0$ (Good)", fontsize=14)
    axs[1, 0].axis('off')
    
    # (1, 1) Solid w/ Border (差几何)
    L_solid = 27
    grid_solid, _ = generate_solid_with_border(L_solid)
    axs[1, 1].imshow(grid_solid, cmap=cmap)
    axs[1, 1].set_title(f"e) Solid w/ Border (L={L_solid})\n$D_{{theory}} = 2.0$ (Bad)", fontsize=14)
    axs[1, 1].axis('off')
    
    # (1, 2) 隐藏第六个子图
    axs[1, 2].axis('off')

    # --- 3. 保存并显示 ---
    
    # --- [修复] 调整 rect 来适应新的 suptitle 位置 ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # <--- 顶部边界从 0.95 调整为 0.93
    
    output_filename = "Figure_1_Geometries.png"
    plt.savefig(output_filename)
    print(f"Successfully saved image: {output_filename}")
    
    plt.show()
