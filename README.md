# A Physical Bottleneck Theory of Scaling Laws for Large Models

This repository contains the source code and experimental data for the paper **"A Physical Bottleneck Theory of Scaling Laws for Large Models"**.

We provide the "digital wind tunnel" simulations used to decouple the hardware geometry ($\delta$) and software efficiency ($\gamma$) bottlenecks in AI scaling.

## 📂 Repository Structure

### 1. Hardware Bottleneck ($\delta \to \alpha$)
* **`src/run_delta_wind_tunnel.py`**:
    * **Purpose**: Simulates thermal resistance ($R_{th}$) for 5 different geometries (Vicsek, Sierpinski, Chessboard, etc.).
    * **Output**: Proves the $R_{th} \propto V_{eff}/A_{eff}$ law.
    * **Corresponds to**: **Figure 1** and **Figure 2** in the paper.

### 2. Software Bottleneck ($\gamma \to K$)
* **`src/hamster_room_simulation.py`**:
    * **Purpose**: Multi-particle simulation to measure crowding ($\rho$) under different pressures ($S$).
    * **Output**: Reveals the "Gas-Liquid-Jammed" phase transition.
    * **Corresponds to**: **Figure 3** (Phase Diagram).

* **`src/gamma_rho_pde.py`**:
    * **Purpose**: PDE hotspot simulation to map crowding ($\rho$) to efficiency ($\gamma$).
    * **Output**: Fits the non-linear collapse function $\gamma \approx 1/(1+k\rho)$.
    * **Corresponds to**: **Figure 4**.

### 3. Strategic Validation (Toy Network)
* **`src/train_toy_network.py`**:
    * **Purpose**: Trains Dense vs. MoE MLPs on a regression task to measure $K$ values.
    * **Output**: Confirms MoE has a significantly lower $K$ (better starting point) due to low crowding.
    * **Corresponds to**: **Figure 5**.

## 🚀 How to Run

Dependencies: `numpy`, `matplotlib`, `scipy`, `torch` (for toy network).

```bash
# Run the hardware geometry experiment
python src/run_delta_wind_tunnel.py

# Run the phase transition simulation
python src/hamster_room_simulation.py
```
