# 论文与 README 绘图规范

本规范用于本仓库后续新绘制或重新绘制的科研结果图、物理场图和架构示意图。
目标是：**简洁、可比较、可复现；论文 PDF 中的文字清晰且可编辑。**
旧图不因本文发布而自动视为合规，重绘时再逐项执行。

风格参考用户提供的 `eval.ipynb` 和 `eval (1).ipynb`：Helvetica、NPG 配色、
细轴线、向内刻度、无边框图例、紧凑多面板、等比例物理场。
其中旧 notebook 的部分 9 pt 标签、默认数学字体和逐面板自动色限不直接沿用；
以本文的 5–7 pt、无衬线公式和统一比较色限为准。

## 1. 必须满足的要求

| 项目 | 要求 |
|---|---|
| 正文字体 | Helvetica；显式记录并加载实际字体文件路径 |
| 数学字体 | 与 Helvetica 协调的无衬线数学字体；不得混入默认衬线公式 |
| 字号 | 最终论文排版尺寸下 **5–7 pt**，不得通过缩图使文字小于 5 pt |
| 文字颜色 | 黑色或深灰；不使用彩色标题、图例文字、数值或公式 |
| 分类图配色 | 曲线、散点、箱线图和小提琴图使用固定 NPG `nrc` 十色色板 |
| 图内文字 | 只留变量、单位、方法、时间、面板标号等必要短标签 |
| 论文输出 | PDF，嵌入 TrueType 字体，保留文字和可编辑矢量对象 |
| 禁止事项 | 文字转曲、整图栅格化、以截图或 PNG 套入 PDF 代替矢量输出 |
| README 输出 | 同内容、同布局的 PNG 预览；它不是可编辑的论文原稿 |
| 可复现性 | 保留绘图源码、直接数据路径、配置和样本/时间选择记录 |

## 2. 字体、公式与最终字号

### Helvetica 的获取与加载

1. 先检查已有字体，包括 Regular、Bold 和实际需要的 Oblique 字体。
   不仅检查 family 名，还要核对字体元数据及实际解析到的文件路径。
2. 如果没有 Helvetica，从已有授权资源或
   [Helvetica 官方销售渠道](https://www.myfonts.com/collections/helvetica-font-linotype)
   获取并下载可嵌入的桌面字体，记录来源。字体文件保存在机器的字体目录，
   不把受限字体二进制提交到公开仓库。
3. 绘图参数显式给出字体的绝对路径，例如通过
   `PLOT_HELVETICA_REGULAR`、`PLOT_HELVETICA_BOLD` 传入。
   每次进程启动都注册所需字体，按需用 `FontProperties(fname=...)` 指定。
4. 缺文件、字形缺失、解析到不同字体或无法嵌入时，明确报错。
   不静默替换为 Arial、Liberation Sans、Nimbus Sans 或 DejaVu Sans 后仍称为 Helvetica。
   缺少可用授权文件时报告这个具体缺项；不把未安装字体写成已经就绪。

下面仅示意路径检查；真实绘图还须登记其他使用到的字重和数学字体路径：

```python
import os
from pathlib import Path
from matplotlib import font_manager

font_path = Path(os.environ["PLOT_HELVETICA_REGULAR"]).resolve(strict=True)
font_manager.fontManager.addfont(str(font_path))
font = font_manager.FontProperties(fname=str(font_path))
family = font.get_name()
resolved = Path(font_manager.findfont(
    font_manager.FontProperties(family=family), fallback_to_default=False
)).resolve()
if resolved != font_path:
    raise RuntimeError(f"Unexpected font resolution: {resolved}; expected {font_path}")
```

### 无衬线公式

默认采用 Matplotlib Mathtext，`text.usetex=False`、
`mathtext.fontset="dejavusans"`：正文保持 Helvetica，公式使用明确指定的
DejaVu Sans 数学字形。二者不冒充同一字体；实际用到的数学字体文件也要记录。
变量用斜体，单位和函数名用正体；避免为短标签引入多层分式或复杂下标。
设置 `mathtext.fallback=None`，并检查缺字警告和 PDF 字体清单，防止未声明的
Computer Modern、Times/STIX 衬线字形混入。
如选择 Helvetica 自定义数学字体或 LaTeX 路线，必须单独验证希腊字母、上下标、
积分符号和最终嵌入字体，不能仅设置正文字体就声称公式已统一。
参见 [Matplotlib 数学字体配置](https://matplotlib.org/stable/users/explain/text/mathtext.html#fonts)。

### 字号和图幅

先确定论文中实际占用的宽高，再设置 `figsize`。暂未指定期刊模板时，
单栏以 85 mm 宽作为工作尺寸；大幅多面板图按用户确认的 **170 mm** 正文宽度，
对应 A4 左右各 20 mm 页边距。这不是所有期刊的通用要求。
宽度换算为英寸时除以 25.4。直接在目标尺寸排版并导出，避免插入 LaTeX 后
再整体缩放而改变字号。模板正文宽度不同时，应先确认再重排。

| 元素 | 建议字号 |
|---|---:|
| 面板标号、轴标签、短标题 | 7 pt |
| 刻度、图例、色条刻度、样本/时间标签 | 6 pt |
| 少量次要注释 | 5.5–6 pt |

5 pt 是下限，不是默认值。全部文字（包括图例标题、科学计数法偏移量、
色条和公式上下标）均需检查。Mathtext 会缩小上下标；必要时使用 7 pt 的
公式标签、简化写法或移到图注，不允许靠过小公式塞进面板。
最终 PDF 按 100% 尺寸检查可读性；在屏幕上放大后可读不能代替这一检查。

## 3. 分类图配色

### 分类曲线、散点、箱线图和小提琴图

使用与两个 notebook 一致的 NPG `nrc` 十色，顺序固定：

```python
NPG = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]
```

它是 ggsci 的分类色板，不是期刊强制指定的连续色图。
来源：[ggsci NPG palette](https://nanx.me/ggsci/reference/pal_npg.html)。
直接固定十六进制值即可，不要求为了取色而引入 ggsci 或 Seaborn 依赖。

同一组图中方法颜色必须一致。新 nonlocal/local-AD 比较默认使用：

| 方法 | 颜色 | 线型 |
|---|---|---|
| Nonlocal | `#E64B35` | 实线 |
| Local AD | `#4DBBD5` | 实线（收敛图） |
| FEM PINO | `#00A087` | 实线（收敛图） |
| Reference（曲线/轮廓） | `#333333` | 点划线 |

其余方法按 NPG 顺序扩展，在绘图配置中记录映射；不依赖某张图的临时绘制顺序。
收敛图按下述已确认模板统一实线；需要辅助辨认时可在实际观测点使用不同标记，
不把平滑网格当作实测点。其他比较图可用线型/标记辅助辨认。
所有图例文字仍用黑色或深灰。

## 4. 布局和信息密度

- 白色背景，细轴线和刻度线（默认 0.5 pt），向内刻度，无边框图例。
  数据线约 0.8–1.0 pt，标记约 2–3 pt；不使用装饰阴影、渐变背景或厚边框。
- 图内使用简短英文标签，如 `Reference`、`Nonlocal`、`Local AD`、
  `Abs. error`、`Rel. L² Error`。长解释、训练配置和结论放在图注或 README。
- 图内不再添加算例名称作为标题；算例名写入 PDF/PNG 文件名，由 README 或论文图注说明。
- 面板标号使用统一的 `(a)`、`(b)`；同一组只显示一次方法图例、时间列标题
  和单位。不在每个小图重复完整标题和误差解释。
- 物理场保持真实长宽比。多样本演化图通常展示 3–5 个样本和 3–5 个实际保存时刻。
  按样本分组、按时间排列，方法行顺序一致。放不下就拆图，不缩小文字。
- 同组比较中，同一变量使用一致色限，误差场单独统一色限；不通过逐面板
  自动拉伸或隐藏超范围值改变视觉结论。使用不同色限的补充局部图须明确标注。
- 网格场允许平滑显示，但不得跨孔洞插值、改变几何或掩盖未解析界面。
  指标仍从声明的原始评估数组计算；明确显示插值与数值求解精度的区别。
- 误差图必须说明计算的是原始节点差值的显示，还是显示插值后的差值。
  3D 场采用一致相机、比例和光照；色条、坐标和文字单独矢量叠加。
- 示意图应标注为 setup/schematic，与真实求解结果区分；文字和箭头保持可编辑。

## 5. 统计图与误差口径

绘图不得悄悄修改指标。当前总体误差约定是：对样本 \(i\)、物理通道 \(k\)，
先在该样本完整时空评估点集 \(Q_i\) 上分别计算分子和分母，再平均通道：

$$
e_{i,k}=
\frac{\sqrt{\sum_{q\in Q_i}|\hat u_{i,k}(q)-u_{i,k}(q)|^2}}
     {\sqrt{\sum_{q\in Q_i}|u_{i,k}(q)|^2}},
\qquad
e_i=\frac{1}{C}\sum_{k=1}^{C}e_{i,k},
\qquad
\bar e=\frac{1}{N}\sum_{i=1}^{N}e_i.
$$

时间依赖问题覆盖完整时间区间；静态问题覆盖空间域。使用物理输出通道，
不默认加入网络潜变量或辅助输出。记录具体通道、坐标、时间和样本数。
当前约定为评估点等权，不是几何体积积分；若另给求积加权指标，应另行命名。
参考通道零范数须明确报告和另定度量，不偷偷加 epsilon 改变相对误差定义。

- 分布图保留 \(N\) 个 \(e_i\)，只在通道维度求平均；总体数值再对样本平均。
  不把样本/通道拼起来求一次范数，也不平均逐时刻相对误差代替全时空范数。
- 当前两个导数对照使用一个严格配对 seed 的最终 checkpoint、64 个样本。
  后续图必须记录实际 seed、最终步数和共同测试集；不能混入其他 seed 当作额外样本。
- 小提琴图采用紧凑布局、填色和细轮廓，用内部虚线表示第 25、50、75 百分位，
  默认不叠加散点、均值文字或粗黑区间条。当前两个算例分别输出两张图。
  全部样本均参与 KDE；不显示散点不等于删除离群样本。允许密度尾部超出样本极值，
  不在最小/最大观测值处硬截断；需记录尾部延伸设置，并说明它不代表新增样本。
  固定并记录带宽及密度归一化方式；图注说明 KDE 只是有限样本分布的平滑展示。
- 若画收敛曲线，仅使用实际记录/评估的步数。不从少量 checkpoint 编造密集曲线。
  默认沿用下述已确认的局部平滑模板，保留原始曲线，并标明拟合与实测的区别。
  如果另画均值±标准差，必须注明是样本间标准差，不能叫置信区间。
- 物理场面板下方的误差采用乘号与上标指数形式，如 \(1.25\times10^{-2}\)，
  不用 `1.25e-02`；上标也须满足 5 pt 下限，并保留可编辑文字。
- 误差图直接绘制原始相对误差比值，**不乘以 100，纵轴标签不加 `%`**。
  当前小提琴图使用对数纵轴，在对数误差空间估计 KDE，避免尾部延伸到负误差；
  图注注明密度的计算空间，轴刻度仍显示原始比值。纵轴范围交给 Matplotlib
  自动决定，不手动设置 `ylim`。对数轴要求正值；遇到零值须明确处理，不能静默删除。

### 后续默认模板（2026-11-14 用户确认）

以已接受的两相流对照图为样式参考：
[收敛图](../assets/pino-studies/channel_nsch2d_fem_convergence.png)、
[分布图](../assets/pino-studies/channel_nsch2d_fem_distribution.png)。
多面板图按170mm实际宽度排版；面板数按展示内容调整，不固定为四个物理场。
不改变各算例已声明的误差通道、测试集、最终权重或统计口径。

**收敛图**：各方法均用实线；浅色细线显示全部原始记录，同色粗线显示局部
LOWESS趋势，透明填充显示适用的95%区间。当前参考样式为原始线宽0.55pt、
透明度0.35，平滑线宽1.05pt，区间透明度0.18；无需每张图重复方法之外的长图例。
正误差可在log10空间做局部线性拟合，再转换回原始误差轴。显示用的密集拟合
网格不是额外checkpoint；最终误差、表格和结论仍使用未平滑的数值。

单次训练记录有先后相关性时，可参考
[`scripts/local_trend.py`](../scripts/local_trend.py) 的循环分块残差bootstrap，
并明确标注为**平滑趋势的近似逐点95%置信区间**。它依赖平滑趋势和近似平稳、
短记忆残差假设，不代表跨seed、测试样本或整条曲线的联合95%覆盖，也不覆盖
未知的平滑偏差。当前两相流的41条记录采用span0.22、4记录一块、2,000次重采样、
bootstrap随机种子20261114；这些是此图的复现参数，不要求其他算例机械照搬。
必须根据记录数量、间隔和相关性说明所选参数；数据太少或假设明显不适用时，
保留原始记录并明确区间缺项，不为填色编造置信区间。若有独立重复训练，可另用
重复实验估计均值置信区间，但须记录重复数、重采样单位和统计方法。

**分布图**：默认采用当前小提琴样式；NPG填色、深灰细轮廓、内部25/50/75%
分位数虚线，无额外散点、粗黑条或均值文字。默认在log10误差空间使用Gaussian
KDE、Scott带宽，尾部延伸3个带宽，各小提琴按相同最大宽度归一化；当前描边为
0.6pt、填色透明度0.85。所有实际样本参与，不在样本极值处硬截断尾部；平滑尾部
不表示新增观测。图注注明样本数，极小样本仅作描述展示，不能据此声称总体密度
已得到可靠估计。零值或相同值导致对数/KDE无定义时须明确处理，不静默删点或加扰动。

对应渲染源码为
[`scripts/render_channel_fem_comparison.py`](../scripts/render_channel_fem_comparison.py)，
完整配置及直接数据路径在
[`artifacts/channel_fem_comparison_figures.json`](../artifacts/channel_fem_comparison_figures.json)。
同一组图的PDF、PNG、README与网站图注一并更新；旧图无需仅为风格统一而全部重绘。

## 6. 导出与可编辑性

PDF 是论文主文件，PNG 是 README 预览，二者使用同一个绘图对象和内容。
优先保存矢量曲线、边框、轮廓和文字。密集热图/三角网格/3D 渲染可以仅栅格化
场图层（建议最终尺寸下 300–600 dpi），避免超大 PDF；文字、刻度、色条标签
不可随场图层栅格化。PNG 默认 300 dpi，优先优化布局和压缩以满足仓库体积预算。

必须显式设置 `pdf.fonttype=42`、`ps.fonttype=42`、`pdf.use14corefonts=False`。
这里的 42 是 Matplotlib 的 TrueType 输出选项；PDF 检查工具可能显示
`TrueType` 或 `CID TrueType`（对应 CIDFontType2），不要求字面显示 `Type 42`。
禁止 Type 3 字体、缺失嵌入和用字形轮廓冒充文本。
可选 SVG 设置 `svg.fonttype="none"`，但它仍依赖查看机器的字体，不能代替嵌入字体的 PDF。
参见 [Matplotlib PDF/PS 与 SVG 字体说明](https://matplotlib.org/stable/users/explain/text/fonts.html)。

在 Illustrator 等软件保存时，启用 **Preserve editing capabilities / 保留编辑功能**，
不执行 Outline text / Create outlines，也不扁平化文字。Matplotlib 没有同名开关，
需通过保留文本对象、嵌入字体和实际编辑检查满足要求。
字体嵌入不保证每个编辑器均能无损编辑：PDF 可能嵌入字体子集，编辑机器仍需安装
相同字体。必须同时保留绘图源码；复杂公式的原始表达式也要保存。

建议的基础设置（字体路径验证后、创建图之前应用；额外 style/theme 不得覆盖）：

```python
import matplotlib as mpl
from cycler import cycler

STYLE = {
    "font.family": "sans-serif", "font.sans-serif": [family],
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "figure.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "legend.title_fontsize": 6,
    "text.color": "#222222", "axes.labelcolor": "#222222",
    "axes.titlecolor": "#222222", "xtick.color": "#222222",
    "ytick.color": "#222222", "legend.labelcolor": "#222222",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 0.9, "lines.markersize": 2.5,
    "axes.grid": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.transparent": False,
    "axes.prop_cycle": cycler(color=NPG),
    "text.usetex": False, "mathtext.fontset": "dejavusans",
    "mathtext.fallback": None,
    "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False,
    "svg.fonttype": "none", "figure.dpi": 150, "savefig.dpi": 300,
}
```

使用 `mpl.rc_context(STYLE)` 创建并导出图；方法配色仍按第 3 节显式赋值。
`bbox_inches="tight"` 会改变最终页面尺寸：使用后必须重新核对 PDF 宽高及论文
缩放比例，不能只根据原始 `figsize` 判断最终字号。不要为了紧凑而裁掉上下标或图例。

## 7. 文件组织与复现记录

- README 精选 PNG 继续放在已有 `assets/` 对应子目录，文件名包含算例名及图的用途。
  论文 PDF 默认保存到相应外部实验/绘图产物目录，记录直接路径。
  小型精选 PDF 如需入库，应符合现有仓库文件预算，避免批量提交重复版本。
  具体预算见 [assets/README.md](../assets/README.md) 和
  [DEVELOPMENT.md](DEVELOPMENT.md)。
- 每组图保留绘图脚本或可顺序执行的 notebook，以及一条可复现命令。
  不依赖 notebook 隐藏状态；无需重新训练便能从已有数组重绘。
- 配套 JSON/记录至少写清：数据/权重/配置的直接路径，reference 方法与网格，
  样本 ID、seed、checkpoint 步数、实际时刻、物理通道、误差计算方式，
  色限/插值/KDE 设置，图幅、字体文件路径、软件版本以及 PDF/PNG 输出路径。
  不使用哈希代替文件路径加载；样本和曲线不可按结果好坏事后挑选。
- PDF 与 PNG 改版同步；原始数值与既有实验结论不因美化而更改。
  原始数组、权重、完整实验图库和字体文件不随精选图片一起批量入库。

## 8. 每次交付前检查

1. 最终排版尺寸下全部文字在 5–7 pt，清楚、黑/深灰、无重叠或裁切。
2. PDF 中正文、刻度、色条标签、图例和公式仍是文本对象，没有转曲或被栅格化。
3. 执行 `pdffonts figure.pdf`，确认所有字体 `emb=yes`、无 Type 3、无意外替代字体；
   使用 `pdftotext figure.pdf -` 抽查标签可提取。提取成功不是编辑性的充分条件，
   还需在目标 PDF/矢量编辑器中试改一个普通标签和一个公式标签。
4. 在最终尺寸看整体，在放大视图检查符号、色条和边界；已声明的场图层可为位图。
5. 比较图配色、色限、时间、几何比例一致；所有统计值与保存的计算记录一致。
6. PDF 与 PNG 内容一致，源码、路径记录和生成命令齐全；检查文件体积及 Markdown 链接。

本文件制定阶段只登记规范，不表示已替历史图片完成上述字体/编辑器验收。
