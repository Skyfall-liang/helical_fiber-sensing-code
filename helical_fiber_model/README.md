# 螺旋光纤理论模型与应变解耦

已根据指定论文第三节 A、B 和现有 MATLAB HelixTactileSensor.m 重构。用于每个 PCC 段的弯曲角、偏转角、接触残差、接触候选位置与法向力估计。内部一律使用 m、rad、1/m、N、无量纲 strain；CLI 可读取 με。

## 在 VS Code 运行

1. 用 VS Code 打开本文件所在的 `helical_fiber_model` 文件夹。
2. Python: Select Interpreter 选择 `.venv\Scripts\python.exe`。本机已经准备此环境，并验证运行；它继承本机 NumPy/Matplotlib，SciPy 位于工程 `.vendor`，由环境中的 `.pth` 引入。不要只搬动 `.venv` 到另一台电脑。
3. 打开 `run.py`，F5 选择 **Synthetic demo**，可在 `model.py` 的 `fit()` 中打断点。
4. 实测文件选择 **Fit CSV**（默认输入单位为 με），或运行下面命令。

```powershell
.\.venv\Scripts\python.exe run.py --show
.\.venv\Scripts\python.exe run.py --input data/bending_contact.csv --unit microstrain --out outputs/real --show
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

新电脑/重新建立独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

需要 VS Code 的 Python 和 Python Debugger 扩展。`--show` 打开图窗；省略时只保存图。`--method ols` 可对照普通最小二乘，默认 `tukey`。

## 文件入口

- `helical_fiber/model.py`：参数、正向应变、Tukey IRLS、峰检测、分区力标定、PCC 三维坐标。
- `run.py`：仿真/CSV 入口，导出 JSON、CSV、PNG。
- `config/segment1.json`、`segment2.json`：论文两段的力标定，独立使用每段局部光纤坐标。
- `data/`：从现有 MATLAB 文件转换的两组实测样例，来源见 `provenance.json`。
- `docs/model_notes.md`：公式推导、论文/MATLAB 差异、参数物理意义与使用边界。
- `docs/validation.md`：已执行的数值测试和实测处理结果。

每次运行生成 `result.json`（角度、曲率、接触候选、诊断），`decoupled.csv`（弯曲/残差/权重），`input.csv`（规范化的无量纲输入）和 `analysis.png`。

## 实际数据格式

CSV 表头必须是 `s_m,strain`，每行一个光纤位置，单文件表示一个采样时刻/一帧。示意：

```csv
s_m,strain
0.000,12.4
0.001,13.2
0.002,14.1
```

示意值单位为 με，应传 `--unit microstrain`。实际拟合需要至少 10 个有效点且足够的相位覆盖。`s_m` 是敏感段起点起算的光纤弧长，须递增，范围 `[0,Lf]`；不能直接填主干轴向坐标或原始全长光纤坐标。引线须先裁去并减掉敏感段坐标起点。不要每帧改变坐标零位。缺失 strain 可写 NaN，拟合会跳过，不填零。

示例文件沿用 MATLAB `linspace(0,4.277,N)`，并非对其物理坐标做了重新标定。原始 ODiSI TSV/实时 TCP 尚未接入；先导出上述单帧 CSV，或直接调用 Python API。

## Python API

```python
import numpy as np
from helical_fiber import Config, HelicalFiberModel, calibrate_force

model = HelicalFiberModel(Config())
s = np.linspace(0, model.config.fiber_length_m, 6395)
# 曲率 = 弯曲角(rad) / 主干长度(m)
kappa = np.deg2rad(30) / model.config.backbone_length_m
sim = model.simulate(s, kappa, np.deg2rad(40), contacts=[
    {"s_m": 1.6, "force_n": 2.0, "width_m": .01}
])
result = model.fit(s, sim["strain"])
print(result["summary"], result["contacts"])

# 每段、每种结构区分别标定；这里输入的是接触残差峰，不是总应变峰。
cal = calibrate_force([1, 2, 3], np.array([60, 80, 100])*1e-6)
```

正向高斯接触核是仿真假设；论文只规定残差检测和峰值-力标定，并未推导完整接触应变场。`width_m` 沿光纤计量，不等于轴向接触宽度。

## 实验接入前应确认

- `strain_scale` 与已知弯曲角标定：模型仅观测到 Cscale×κ，不能同时无先验求两者。
- `phase_zero_rad`：应变模型的固定相位零位。`geometry_phi_offset_rad`：拟合相位到实体弯曲平面方位的零位差，默认跟随 MATLAB 直接使用 φ。应通过已知弯曲方向核对。
- 半径、螺距、敏感光纤长度、采样坐标与实物一致。
- 分区力系数与实际装置一致；论文标定范围为 1–3 N，超出会标记 `calibration_extrapolated`。
- 相邻圈的多个峰是接触候选，不自动等于多个独立外力，不应直接相加当作总力。

当前实测样例没有配套角度/力/位置真值，已完成流程验证，尚不能宣称真实装置反演准确度达到论文指标。
