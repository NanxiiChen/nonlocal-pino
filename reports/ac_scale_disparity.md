# AC方程空间-时间尺度差异问题

## 问题

在 KKS 相场腐蚀模型中，非局部 Allen-Cahn 方程为：

$$\frac{\partial\phi}{\partial t} + N_{AC3} \cdot L_\delta[\phi] + N_{AC2} \cdot g'(\phi) - N_{AC1} \cdot (c - h(\phi)\Delta c_0 - c_{Le}) \cdot \Delta c_0 \cdot h'(\phi) = 0$$

其中无量纲系数：

| 系数 | 表达式 | 扩散控制 (L=2.0) | 激活控制 (L=1e-11) |
|---|---|---|---|
| $N_{CH}$ | $D T_c / L_c^2$ | $4.25 \times 10^0$ | $8.5 \times 10^3$ |
| $N_{AC1}$ | $2 A L T_c$ | $1.07 \times 10^{10}$ | $1.07 \times 10^2$ |
| $N_{AC2}$ | $L \omega_\phi T_c$ | $1.76 \times 10^9$ | $1.76 \times 10^1$ |
| $N_{AC3}$ | $L \alpha_\phi T_c / L_c^2$ | $1.03 \times 10^6$ | $1.03 \times 10^{-2}$ |

## 诊断数据 (扩散控制, training step ~4600)

在 FEM 网格上评估 PINN 预测的 AC 残差各分量 RMS：

| 分量 | RMS | 占残差比 |
|---|---|---|
| $\partial\phi/\partial t$ | **1.86** | ~10⁻⁸ |
| $N_{AC3} \cdot L_\delta[\phi]$ | 1.85×10⁸ | — |
| $N_{AC2} \cdot g'(\phi)$ | 8.29×10⁷ | — |
| AC driving (含 $N_{AC1}$) | 5.32×10⁶ | — |
| **总残差** | 1.43×10⁸ | 100% |

## 根因

1. **空间项系数本身极大**：$N_{AC1} \sim 10^{10}$, $N_{AC2} \sim 10^9$, $N_{AC3} \sim 10^6$。这些是材料固有参数 (A ~ 10⁷ J/m³, ω_φ ~ 10⁷ J/m³)，无法通过无量纲化消除——选择不同的 $T_c$ 只会等比例改变所有项，不改变相对比例。

2. **空间项不完全抵消**：理论上 $\phi$ 和 $c$ 耦合使三个空间项之和 ≈ $-\partial\phi/\partial t$。但实际上它们的 RMS 在 10⁶–10⁸ 量级，彼此仅部分抵消，残差仍为 10⁸ 量级。

3. **时间导数被淹没**：$\partial\phi/\partial t$ 的 RMS 仅 ~2，比空间项小 8 个数量级。在 MSE loss 中：
   $$\mathcal{L}_{AC} = \text{mean}\big((\partial_t\phi + \text{spatial\_terms})^2\big)$$
   $\partial_t\phi$ 贡献 $\sim 4$，空间项贡献 $\sim 10^{16}$。loss 的值几乎完全由空间项决定。

4. **梯度方向被绑架**：对网络参数 $\theta$ 的梯度：
   $$\frac{\partial\mathcal{L}_{AC}}{\partial\theta} = 2 \cdot r_{AC} \cdot \frac{\partial r_{AC}}{\partial\theta}$$
   其中 $\partial r_{AC}/\partial\theta$ 包含 $N_{AC1} \sim 10^{10}$ 等系数作用于空间项的梯度分量。虽然 $\partial_t\phi$ 的梯度也存在（经过 $\mathbf{F}_x \cdot \partial\mathbf{F}_t/\partial\theta_t$ 传回 temporal branch），但其量级仅是空间梯度的 $10^{-9}$ 倍。optimizer 看到的最陡下降方向永远是"调整空间平衡"，不是"让界面演化"。

5. **Landscape 存在浅谷**：任何满足"空间项近似平衡"的 $(\phi, c)$ 都是低 loss 解，而界面速度是不受约束的自由参数。正确的时变解只是众多低 loss 解之一，且不是最深的盆地。

## CH 方程对比

Cahn-Hilliard 方程的非局部形式：

$$\frac{\partial c}{\partial t} + N_{CH} \cdot L_\delta[c] - N_{CH} \cdot \Delta c_0 \cdot L_\delta[h(\phi)] = 0$$

| 控制机制 | $N_{CH}$ | $\partial c/\partial t$ 典型 RMS | 时间/空间尺度比 |
|---|---|---|---|
| 扩散 (L=2.0) | 4.25 | ~1.6 | ~0.4（**基本平衡**）|
| 激活 (L=1e-11) | 8.5×10³ | ~1 | ~10⁻⁴（**4 个数量级差异**）|

扩散控制下 CH 方程无显著尺度问题。但激活控制下 $N_{CH} \sim 8500$，空间项比时间导数大 ~5000 倍。激活控制时 **AC 和 CH 同时存在尺度差异**，AC 更严重（10⁸ vs 10³）。

## 后果

- PINN 预测的界面移动速度可任意偏离真实解
- rel L2 error 在训练中从 38% 恶化为 40%（越训越差）
- 空间残差仍然巨大（1.4×10⁸），说明网络未找到精确的空间平衡
- optimizer 在"空间平衡 + 任意速度"的平坦方向上漂移

## 尝试过但未解决的问题

| 方法 | 效果 |
|---|---|
| `ac_scale` 整体缩放 | 只缩放全局因子，不改变梯度方向的比例 |
| GradNorm 自适应权重 | 平衡 AC/CH/BC/IC 之间，不解决 AC 内部梯度失衡 |
| Causal weighting | 强制早期时间先收敛，但静态解同样满足早期时间 |
| Stagger 训练 | 交替训练 AC/CH，不解决 AC 内部 |
| 硬约束输出层 | 确保边界合规，无法驱动演化 |
| Split-PINN 架构 | 有足够 rank 表达能力，不是架构瓶颈 |
| SOAP 优化器 | 改善 Hessian 条件数，但 10⁸ 差距过大 |

## 可行方案

### 时间推进 (Time-marching curriculum)

将时间域分段，逐步扩展：
- Stage 1: t ∈ [0, 0.05]，训练至收敛
- Stage 2: t ∈ [0, 0.1]，用 Stage 1 的参数初始化
- ...
- Stage N: t ∈ [0, 1.0]

每阶段 optimizer 只需要在已有近似解上微调，不容易漂移到静态解。

### 逐项归一化 Loss

对 AC 残差的各分量分别计算 EMA 归一化方差，使各项在 loss 中等贡献：

$$\mathcal{L}_{AC} = \frac{\text{MSE}(\partial_t\phi)}{\sigma^2_t} + \frac{\text{MSE}(N_{AC3} L_\delta[\phi])}{\sigma^2_D} + \frac{\text{MSE}(N_{AC2} g')}{\sigma^2_g} + \cdots$$

不改变 PDE 本身，收敛时等价于原方程。

### Mixed form / 辅助变量重构

另一种更接近刚性 PDE 常用处理的方法，是引入辅助变量，将“时间演化”和“刚性空间平衡”拆成两个子问题，而不是继续把所有项压进一个强形式残差里。

对 AC 方程，可定义辅助变量 $q$ 表示全部空间驱动项：

$$q = N_{AC3} L_\delta[\phi] + N_{AC2} g'(\phi) - N_{AC1} (c - h(\phi)\Delta c_0 - c_{Le}) \Delta c_0 h'(\phi)$$

则原方程改写为一阶混合系统：

$$\begin{aligned}
\frac{\partial\phi}{\partial t} + q &= 0 \\
q - \Big(N_{AC3} L_\delta[\phi] + N_{AC2} g'(\phi) - N_{AC1} (c - h(\phi)\Delta c_0 - c_{Le}) \Delta c_0 h'(\phi)\Big) &= 0
\end{aligned}$$

训练时不再只最小化单个残差 $r_{AC}$，而是分别约束：

$$r_1 = \partial_t \phi + q, \qquad r_2 = q - q_{\text{spatial}}(\phi, c)$$

其直觉是：

- $r_1$ 直接约束界面速度，不再让时间导数仅通过与 $10^6$–$10^{10}$ 量级空间项“硬碰硬”来被感知；
- $r_2$ 单独负责空间平衡，可对其单独加权、归一化或 curriculum；
- optimizer 看到的是两个子系统的耦合，而不是一个被巨大系数主导的单残差。

这与“逐项归一化 Loss”不同：后者仍在同一个强形式残差里做数值重标定；mixed form 则显式改变优化问题的结构，把原本依赖彼此抵消的项拆开处理。

在 PINN 实现上，mixed form 有两种层级：

1. **Pseudo-mixed**：网络仍只输出 $(\phi, c)$，但训练时显式构造 $r_1, r_2$ 两个残差并分别加权；
2. **True mixed**：网络额外输出辅助变量 $q$，未知量变为 $(\phi, c, q)$，真正把“速度约束”和“空间驱动约束”解耦。

前者改动小，适合验证“项内梯度失衡是否是主因”；后者改动更大，但更有机会真正改变 landscape。

类似思想在其他领域中较常见：

- Cahn-Hilliard 方程常引入化学势 $\mu$ 作为辅助变量，写成 $(c, \mu)$ mixed form；
- 不可压流常将速度-压力写成 mixed system，避免单方程中高阶导数和约束直接耦合；
- 刚性反应扩散、半导体漂移扩散、弹塑性 return mapping 中，也常通过辅助变量或 operator splitting 降低优化/求解刚性。

对本问题而言，AC 比 CH 更适合优先做 mixed 重构：扩散控制下 AC 的时间-空间尺度失衡最严重，而 CH 在该工况下尚未表现出同等级别的病态。若后续针对激活控制训练，CH 也可仿照标准 mixed CH 写成：

$$\mu = c - h(\phi)\Delta c_0 - c_{Le}, \qquad \frac{\partial c}{\partial t} + N_{CH} L_\delta[\mu] = 0$$

即把 $\mu$ 而非 $q$ 作为 CH 的辅助变量。

### 重构 AC 方程

将 AC 方程改写为梯度流形式：

$$\frac{\partial\phi}{\partial t} = -\frac{\delta\mathcal{F}}{\delta\phi}$$

直接最小化自由能泛函 $\mathcal{F}$ 的空间部分（Deep Ritz 方法），避免显式处理空间项的巨大系数。

## 结论

KKS 相场腐蚀模型在非局部形式下存在固有的**空间-时间尺度差异**问题。空间项系数 $10^6$–$10^{10}$ 与时间导数系数 1 的量级差距导致基于物理残差的 PINN 训练陷入"空间平衡优先、时间演化被忽略"的困境。纯 MSE 形式的物理 loss 不适合直接训练此类方程，需要改变训练策略（时间推进）或 loss 构造方式（逐项归一化、mixed form、能量形式）。
