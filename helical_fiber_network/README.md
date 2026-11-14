# 螺旋光纤：形状与接触联合神经网络

此工程在独立文件夹中实现：生成三类合成数据 → 训练联合网络 → 用理论模型重构应变 → 独立测试与实测推理。已有训练权重，可直接运行。原 `helical_fiber_model` 工程没有修改；这里保存了理论模型快照用于基线和三维坐标计算。

## 先在 VS Code 打开并运行

打开本文件所在文件夹，使用 Python 扩展选择 `.venv\Scripts\python.exe`。F5 选择 **4 Predict real bending + contact** 即可加载已有权重，对实测曲线推理。图像在 `outputs/real_bending_contact/analysis.png`，数字结果在同目录 `result.json`。

F5 菜单还包括生成数据、训练、测试，以及纯接触示例。程序默认保存 PNG，不弹出图窗；在 VS Code 点击 PNG 查看。

```powershell
# 无需重训，直接使用已有权重
.\.venv\Scripts\python.exe predict.py
# 纯接触示例的 CSV 使用无量纲应变
.\.venv\Scripts\python.exe predict.py --input data/synthetic/example_pure_contact.csv --unit strain --out outputs/example_pure_contact
# 实测 CSV 使用微应变
.\.venv\Scripts\python.exe predict.py --input data/reference/pure_bending.csv --unit microstrain --out outputs/real_pure_bending
```

## 数据、训练、评估

```powershell
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe train.py --epochs 35
.\.venv\Scripts\python.exe evaluate.py --split test
.\.venv\Scripts\python.exe evaluate.py --split stress
.\.venv\Scripts\python.exe make_preview.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

生成/训练命令会覆盖对应默认输出目录。保留旧实验时，使用 `generate_data.py --out data/experiment2`，`train.py --data data/experiment2 --out runs/experiment2`；推理和评估通过 `--checkpoint runs/experiment2/best.pt` 选择权重。测试集不用于选择轮次或阈值，best.pt 按验证集损失保存。

本机已建立 CPU 环境并完成 35 轮训练。`.venv` 继承本机 NumPy、Matplotlib；`.vendor` 存放 PyTorch CPU、SciPy 及依赖，通过环境中的 `.pth` 加载。移动工程到新位置或新机器应重建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

训练脚本当前明确使用 CPU，不会自动使用显卡；本次训练约 355 秒。需要 NVIDIA 加速时，应另行安装匹配驱动的 CUDA 版 PyTorch 并增加 device 迁移，当前结果不依赖 GPU。

## 工程结构

| 文件 | 用途 |
|---|---|
| `fiber_net/physics.py` | 双正弦、分区灵敏度、角度转换和固定几何 |
| `fiber_net/data.py` | 三类数据生成、独立随机种子、局部多峰和噪声 |
| `fiber_net/network.py` | 共享 1D CNN、形状/接触范围/力分支、可微物理解码器和损失 |
| `fiber_net/inference.py` | 模型加载、重采样、峰提取、三维坐标和力一致性检查 |
| `fiber_net/theory_reference.py` | 原理论工程的快照，作为 Tukey 基线 |
| `generate_data.py` / `train.py` | 数据生成与训练入口 |
| `evaluate.py` / `predict.py` | 独立评估与单帧 CSV 推理 |
| `runs/joint/best.pt` | 已训练权重：验证集最优第 34 轮 |
| `docs/model_and_data.md` | 模型公式、标签定义、适用范围 |
| `docs/validation.md` | 运行验证结果及理论基线对照 |

## 数据内容

三类数据为 `pure_bending`、`bending_contact`、`pure_contact`，按轮转方式分配类别。生成 4096 帧训练、512 帧验证、768 帧测试、768 帧压力测试，每帧 1536 点，覆盖光纤 0–4.277 m。另提供三类各一条 6395 点 CSV 示范。

`.npz` 内 y/contact/nuisance 的单位均为 **100 με**；例如数值 1 表示 100 με。coef 前两维为 κcosφ、κsinφ，单位 1/m，第三维为同样归一化的应变偏置。force 为 N，support 为带正负号的无量纲局部形状。事件 JSON 保存位置（光纤 m）、力（N）、宽度（光纤 m）、符号和簇 ID。

参考实测数据只用于观察波形模式与无真值推理，不作为训练标签或独立真实精度验证。合成曲线复现双正弦包络和相隔约一个螺旋周期的成簇局部峰；不代表已辨识出硅胶真实接触传递函数。

![三类合成波形与实测参考](outputs/overview/data_examples.png)

## 输入和输出

单帧 CSV 表头必须是 `s_m,strain`，s 为敏感段局部光纤弧长，严格递增且覆盖 [0,4.277] m。用 `--unit strain` 或 `--unit microstrain` 明确单位。较密的均匀输入会先抗混叠重采样。当前网络要求完整帧，不接受 NaN；不是实时 TCP 流解析器。

输出包含 θ、φ、κ、局部接触候选的光纤坐标/主干坐标/当前段基座系 XYZ、神经网络力估计、重构应变与重构误差。θ<0.5° 时 φ 返回 null，因为近直线形状的方向不可辨识。

`decoupled.csv` 中 `local_force_n` 是稠密网络力头输出，**只有检测到的接触候选附近有解释意义**；不能对整列积分或求和。`signed_support` 是局部应变形状系数，不是校准后的接触概率。

`result.json` 还输出 `force_from_peak_n`（用重构接触峰按论文线性标定反算）与 `force_head_minus_peak_n`（二者差值）。若局部 support 峰未达到 1，两个力会不同；较大差异提示形状核/标定不一致，不能仅凭重构误差小就信任力。

当前针对论文第一段的固定几何和力标定。改为第二段或新的实物，需修改参数、重新生成/标定并训练；不能直接沿用本权重。相邻圈的峰是局部有效力，不代表已识别的多个独立接触物体。

## 已观察到的实测局限

在名为 pure_bending 的实测参考文件中，网络于光纤 s≈3.996 m 产生一个接触候选；若该文件确实无接触，则为误检。受力实测中另有一处网络力与峰值标定反算相差约 0.89 N。当前交付是合成训练的研究原型，真实感知准确度尚未建立；完整记录见 docs/validation.md。
