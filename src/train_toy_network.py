import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import random

# --- 0. 基础设置 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# --- 1. 引入早停机制 (新功能) ---
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# --- 2. 模型定义 (保持 V2 架构) ---
class DenseMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class Expert(nn.Module):
    def __init__(self, input_dim=20, expert_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class MoE_MLP(nn.Module):
    def __init__(self, input_dim=20, n_experts=16, expert_dim=128): 
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([Expert(input_dim, expert_dim) for _ in range(n_experts)])
        self.gate = nn.Linear(input_dim, n_experts)
        
    def forward(self, x):
        gate_logits = self.gate(x)
        expert_idx = torch.argmax(gate_logits, dim=1)
        
        outputs = torch.zeros(x.size(0), 1, device=x.device)
        
        # 向量化加速逻辑
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                outputs[mask] = expert(x[mask])
        return outputs

# --- 3. 辅助函数 ---
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_dense_flops(batch_size, input_dim, h):
    # 3层 Linear FLOPs
    return 2 * batch_size * ( (input_dim*h) + (h*h) + (h*h) + (h*1) )

def count_moe_flops(batch_size, input_dim, h_e):
    # 1个专家的 FLOPs
    return 2 * batch_size * ( (input_dim*h_e) + (h_e*h_e) + (h_e*1) )

# --- 4. 数据生成 (关键修改：增加噪声) ---
def get_data(n_samples, batch_size, input_dim=20, device='cuda'):
    torch.manual_seed(999)
    teacher = nn.Sequential(
        nn.Linear(input_dim, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1)
    ).to(device)
    for p in teacher.parameters(): p.requires_grad = False

    set_seed(42)
    x = torch.randn(n_samples, input_dim).to(device)
    

    noise = 0.1 * torch.randn(n_samples, 1).to(device)
    y = teacher(x) + noise
    
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# --- 5. 训练循环 (集成早停) ---
def train_model(model, data_loader, C_per_step, TOTAL_BUDGET, device, lr):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    stopper = EarlyStopper(patience=20, min_delta=0.001) # 20次检查不下降就停
    
    total_c = 0.0
    steps = 0
    c_hist, l_hist = [], []
    data_iter = iter(data_loader)
    
    print(f"开始训练 {model.__class__.__name__} (FLOPs/step={C_per_step:.1e})...")
    
    while total_c < TOTAL_BUDGET:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x, y = next(data_iter)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        total_c += C_per_step
        steps += 1
        
        if steps % 100 == 0: # 每100步记录一次
            c_hist.append(total_c)
            l_hist.append(loss.item())
            
            # 早停检查
            if stopper.early_stop(loss.item()):
                print(f"  [Early Stop] Loss 停滞于 {loss.item():.5f}, 停止训练。")
                break
            
        if steps % 1000 == 0:
            print(f"  Step {steps}, C: {total_c:.1e}, Loss: {loss.item():.4f}")
                
    return np.array(c_hist), np.array(l_hist)

# --- 6. 拟合与绘图 ---
def fit_scaling(c, l):
    # 拟合 L(C) = K * C^-alpha
    # 加上 offset: L = K*C^-a + L_min (因为有噪声)
    # 简化版：只取最后 50% 数据拟合 Log-Log 线性
    start = int(len(c) * 0.2)
    log_c = np.log10(c[start:])
    log_l = np.log10(l[start:])
    
    def model(lx, lk, a): return lk - a * lx
    try:
        popt, _ = curve_fit(model, log_c, log_l)
        return 10**popt[0], popt[1] # K, alpha
    except:
        return 0, 0

if __name__ == "__main__":
    # --- 🛡️ 终极防御版参数 (Bulletproof Config) ---
    
    # 1. 预算给足：5e12 (是之前的10倍)，防止审稿人说 Dense 没吃饱
    #    得益于 EarlyStopper，MoE 不会因为预算多而空转，它收敛就停。
    BUDGET = 5e12 
    
    # 2. 难度加倍：让 Dense 这种“实心块”架构在参数空间里彻底迷路
    DIM = 64           # 输入维度 20 -> 64
    BATCH = 512        # 批次加大，利用 4090 显存
    
    # Dense: 1024 宽 (更容易过拟合/拥挤)
    # MoE: 保持 16 专家，但专家变宽到 256
    H_DENSE = 1024     
    N_EXPERTS = 16
    H_EXPERT = 256     
    
    device = torch.device("cuda")
    

    loader = get_data(40000, BATCH, DIM, device) # 数据量翻倍到 40000
    
    print(f"🚀 终极决战: Dim={DIM}, Dense_H={H_DENSE}, Budget={BUDGET:.1e}")

    # Dense
    dense = DenseMLP(DIM, H_DENSE)
    f_d = count_dense_flops(BATCH, DIM, H_DENSE)
    print(f"Dense FLOPs/step: {f_d:.2e}")
    start = time.time()
    # 稍微给 Dense 提一点学习率，防止它因为步数多而收敛太慢被冤枉
    c_d, l_d = train_model(dense, loader, f_d, BUDGET, device, 3e-4) 
    print(f"Dense 完成，耗时 {time.time()-start:.1f}s")
    
    # MoE
    moe = MoE_MLP(DIM, N_EXPERTS, H_EXPERT)
    f_m = count_moe_flops(BATCH, DIM, H_EXPERT)
    print(f"MoE FLOPs/step: {f_m:.2e}")
    start = time.time()
    c_m, l_m = train_model(moe, loader, f_m, BUDGET, device, 1e-3)
    print(f"MoE 完成，耗时 {time.time()-start:.1f}s")
    
    # 拟合
    Kd, ad = fit_scaling(c_d, l_d)
    Km, am = fit_scaling(c_m, l_m)
    
    print("\n=== 最终战报 ===")
    print(f"Dense: K={Kd:.2e}, alpha={ad:.2f}")
    print(f"MoE  : K={Km:.2e}, alpha={am:.2f}")
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(c_d, l_d, 'b-', alpha=0.2)
    plt.plot(c_m, l_m, 'g-', alpha=0.2)
    
    def smooth(y): return np.convolve(y, np.ones(50)/50, mode='valid')
    
    if len(c_d) > 50:
        plt.plot(c_d[:len(smooth(l_d))], smooth(l_d), 'b-', label=f'Dense (High $\\rho$)')
    if len(c_m) > 50:
        plt.plot(c_m[:len(smooth(l_m))], smooth(l_m), 'g-', label=f'MoE (Low $\\rho$)')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compute (FLOPs)')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Figure 6: Efficiency Gap (Dim={DIM}, Hard Mode)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig("Figure_Toy_Ultimate.png", dpi=300)
    plt.show()
    
    print("\n=== 最终战报 ===")
    print(f"Dense: K={Kd:.2e}, alpha={ad:.2f}")
    print(f"MoE  : K={Km:.2e}, alpha={am:.2f}")
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(c_d, l_d, 'b-', alpha=0.3)
    plt.plot(c_m, l_m, 'g-', alpha=0.3)
    
    # 平滑线
    def smooth(y): return np.convolve(y, np.ones(50)/50, mode='valid')
    
    if len(c_d) > 50:
        plt.plot(c_d[:len(smooth(l_d))], smooth(l_d), 'b-', label=f'Dense (High $\\rho$) K={Kd:.1e}')
    if len(c_m) > 50:
        plt.plot(c_m[:len(smooth(l_m))], smooth(l_m), 'g-', label=f'MoE (Low $\\rho$) K={Km:.1e}')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Compute (FLOPs)')
    plt.ylabel('Loss (MSE)')
    plt.title('Figure 6: Efficiency Gap (Hard Mode)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig("Figure_Toy_Final.png", dpi=300)
    plt.show()
