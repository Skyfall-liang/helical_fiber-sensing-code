# 螺旋光纤理论模型与应变解耦

已根据指定论文第三节 A、B 和现有 MATLAB HelixTactileSensor.m 重构。用于每个 PCC 段的弯曲角、偏转角、接触残差、接触候选位置与法向力估计。内部一律使用 m、rad、1/m、N、无量纲 strain；CLI 可读取 με。

## 在 VS Code 运行

用 VS Code 打开仓库根目录。工作区使用 `.conda\python.exe`，如 VS Code 缓存了旧解释器，用 **Python: Select Interpreter** 重新选择。下面的离线命令从 `helical_fiber_model` 子目录运行。

```powershell
& ..\.conda\python.exe run.py --show
& ..\.conda\python.exe run.py --input data/bending_contact.csv --unit microstrain --out outputs/real --show
& ..\.conda\python.exe -m unittest discover -s tests -v
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
- `live.py`：直接接收 ODiSI TCP，或回放 `receive.py` 保存的 `messages.jsonl`，逐条输出三维主干坐标和接触候选。
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

示例文件沿用 MATLAB `linspace(0,4.277,N)`，并非对其物理坐标做了重新标定。

## ODiSI 实时重构

### Distal 单通道标定

从仓库根目录运行 Channel 2。临时敏感段起点为仪器坐标 1.2 m。直线基线和弯曲参考使用同一次 Tare，拟合窗口为仪器坐标 1.8012–4.5988 m。增益按目测约 40° 的无接触弯曲设置，方位角仍待已知方向对齐。

```powershell
& .\.conda\python.exe .\helical_fiber_model\live.py --channel 2 --unit microstrain --sensor-start-m 1.2 --config .\helical_fiber_model\config\distal_provisional.json --baseline .\helical_fiber_model\config\distal_straight_baseline.csv --plot
```

### Proximal 与 distal 串联显示

从仓库根目录运行。两次 `--segment` 按基座到末端的顺序填写。两段使用独立的直线基线和拟合窗口，图窗显示两段直径 35 mm 的棒体及各自的接触候选。

```powershell
& .\.conda\python.exe .\helical_fiber_model\live.py --unit microstrain --sensor-start-m 1.2 `
  --segment 1 .\helical_fiber_model\config\proximal_provisional.json `
  --segment 2 .\helical_fiber_model\config\distal_provisional.json `
  --segment-baseline 1 .\helical_fiber_model\config\proximal_straight_baseline.csv `
  --segment-baseline 2 .\helical_fiber_model\config\distal_straight_baseline.csv `
  --serial --plot
```

串联图使用 proximal 基座坐标系，假定两段连接处没有额外安装转角。图形使用各通道最新拟合，终端 JSON 仍逐通道输出该节的局部坐标。

接触显示可在启动时选择。`--contact-mode peak` 为默认模式，沿用峰值红点。`--contact-mode area` 将至少两个连续达到残差阈值的有效样点合成一组，并沿棒体表面画出红色范围。缺测或低于阈值的样点会分开相邻组。JSON 保留 `contacts` 峰值结果，并在 area 模式下增加 `contact_groups`，包含样点范围与对应的表面坐标。红色范围表示光纤上测到的接触响应范围，无法直接换算为物理接触面积。

### Proximal 单通道与独立双通道示例

当前只有 proximal 段时，从仓库根目录运行以下单通道命令。`--sensor-start-m 1.2` 使用仪器坐标的 1.2 m 作为敏感段零点，并保留仪器原始测点间距。

```powershell
& .\.conda\python.exe .\helical_fiber_model\live.py --channel 1 --unit microstrain --sensor-start-m 1.2 --plot
```

实机数据若截至 5.436 m，模型末端 41 mm 没有采样，图中会标记整节形状含外推。接触只在已采样范围内检测。

用本次直线和无接触弯曲数据做的 proximal 临时标定如下。从仓库根目录运行，沿用采集标定数据时的 Tare。

```powershell
& .\.conda\python.exe .\helical_fiber_model\live.py --channel 1 --unit microstrain --sensor-start-m 1.2 --config .\helical_fiber_model\config\proximal_provisional.json --baseline .\helical_fiber_model\config\proximal_straight_baseline.csv --plot
```

基线为同一次 Tare 下直线姿态的逐点中位数。标定使用约 104 mm 的应变载波周期、0.35 m 样条结点间距和 65 με 接触残差高度阈值。弯曲增益按目测约 30° 暂定。实时配置 `live_fit_window_m` 取局部光纤坐标 0.3204–2.9178 m，对应仪器坐标 1.5204–4.1178 m。该窗口根据直线、小弯、大弯和换方向录制选出。拟合保留 1.2 m 的坐标零点和整段几何参数，由中段曲率和方位角重构整节，接触只在中段内检测。重新 Tare 或更改采样网格后，需要重新建立基线。

从仓库根目录运行。两路分别使用各自的既有几何配置和基座坐标系，在图中独立显示。

```powershell
& .\.conda\python.exe .\helical_fiber_model\live.py --host 192.168.1.134 --port 50000 --unit microstrain --segment 1 .\helical_fiber_model\config\segment1.json --segment 2 .\helical_fiber_model\config\segment2.json --assume-span-at-base --plot
```

`live.py` 自己建立 TCP 连接。如果此前已启动 `receive.py`，先停止该采集客户端。省略 `--duration` 时持续运行，按 `Ctrl+C` 或关闭图窗结束。已有录制可用 `--replay 路径\messages.jsonl` 替代连接，回放按文件顺序尽快处理。

默认从 metadata 的 `All Gages` 读取数组起始索引和点数，用 `gage pitch (mm)` 生成每段局部光纤坐标。当前录制通道 1 取索引 4 起的 32 点（80.6 mm），通道 2 取 33 点（83.2 mm），均不足原模型的一圈（约 107 mm）。`--assume-span-at-base` 暂时将这些窗口的首点视为对应节敏感光纤的零点，按真实 2.6 mm 间距拟合。输出将整节形状标记为外推，接触搜索只覆盖有测量的光纤窗口。缺少这个选项时，采样范围与几何配置不符会被拒绝。带 `--fiber-length` 的旧式单通道调用仍可手动指定已确认的均匀采样范围和裁剪索引。

每条 JSON 对应一个通道的一次测量，包含弯曲角、曲率、主干三维坐标，以及接触候选的原始样点索引、局部坐标和有符号残差峰。默认省略接触力，只有实测力标定可用时才加 `--force-calibrated`。两路独立更新，接触候选和绝对位置的准确度取决于实物标定。实时 `--plot` 模式持续接收 TCP 数据，约每 0.2 秒拟合各通道最新一帧。需要逐帧输出时省略 `--plot`。

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

powershell code:
 & .\.conda\python.exe .\helical_fiber_model\live.py --unit microstrain --sensor-start-m 1.2 `
>>   --segment 1 .\helical_fiber_model\config\proximal_provisional.json `
>>   --segment 2 .\helical_fiber_model\config\distal_provisional.json `
>>   --segment-baseline 1 .\helical_fiber_model\config\proximal_straight_baseline.csv `
>>   --segment-baseline 2 .\helical_fiber_model\config\distal_straight_baseline.csv `
>>   --serial --plot --contact-mode area