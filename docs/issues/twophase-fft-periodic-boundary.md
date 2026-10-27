# 两相相场算例：FFT 一阶导数的周期边界限制

状态：**暂缓处理**（2026-10-27，用户明确要求）。仅在用户明确恢复
这项工作时继续研究，不作为当前任务的阻塞项。

## 当前决定

推荐的守恒涡流配置继续采用：

- 空间一阶导数：FFT 计算相场与速度的通量散度，advection=spectral_flux。
- 空间二阶导数：现有 nonlocal pair Laplacian。
- 时间导数：JVP；三次输出映射默认关闭。

当前仍是给定速度场的 CH 相场输运，没有求解 NS 方程。全空间 nonlocal
配置保留为显式实验入口，暂不继续优化或替换默认方案。

## 已知限制

当前标准 Fourier 求导在均匀周期网格上首尾相接，不能直接沿用于一般
非周期壁面、混合边界或任意几何。给网络添加非周期边界硬约束，并不能
改变 Fourier 算子的周期延拓假设。

现有 nonlocal 实现同样使用周期邻域绕回，尚未解决一般边界处理问题。
不能将切换到该实验选项描述为已经支持非周期边界。

本轮仅记录限制；非周期边界的算子、网络和匹配 reference 设计均暂缓。
已有周期算例的配置、数据、checkpoint 和结论保持原样。

## 依据与入口

- [当前算例说明](../../examples/twophase2d/README.md)
- [FFT 通量散度](../../examples/twophase2d/advection.py)
- [周期 nonlocal 算子](../../examples/twophase2d/operators.py)
- [完整一阶算子对照](../tasks/archive/twophase-nonlocal-flux.md)：当前核与
  horizon 下，全 nonlocal 的共同局部参考误差更大，未显示用时优势；
  这不排除其他一阶算子设计，但不据此自动开展后续研究。
