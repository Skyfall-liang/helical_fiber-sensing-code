# 公式、实现约定与来源核对

## 来源

论文：`C:/Users/ZRF/Sync Devices Sharing/2025 Paper/2025-2026 icra/1 Submission/2026_tmech_runfengzhu_Paper Writing.pdf`，重点为第 3–5 页式 (1)–(16)，第 7 页表 III。
MATLAB：`Code_Helix Sensor/Helix sensor streaming and force reconstruction/HelixTactileSensor/HelixTactileSensor.m`、`main_test_Helix.m`、`Calibration_c_k.m`。同时检查了 `AmplitudeModulation.m`、`AmplitudeModulation_2.m` 的包络探索脚本；后者是不同的经验拟合形式，未替代论文最终模型。

## 双正弦、弯曲角与偏转角

设 s 为单段光纤弧长，Lf=4.277 m，主干 L=0.2 m，r=0.017 m，h=0.005 m，N=8，λ=sqrt(r²+(h/2π)²)。

```
z = (L/Lf) s
q(s) = (ω/λ)s + phase_zero
Ψ(s) = sin(π N s/Lf) (-1)^[floor(N s/Lf)+1]
     = -|sin(π N s/Lf)|
ε_b(s) = -Cscale κ r Ψ(s) sin(q(s)+φ)
ε_m(s) = ε_b(s) + ε_contact(s) + offset + noise
θ = κL
```

这里是两个 sin 的乘积（调幅），不是两个独立正弦的相加。符号翻转因子与包络正弦的符号抵消后，Ψ 实际恒为非正；不能再额外翻转载波，否则 φ 会差 π。κ≥0，φ 以 atan2 表示在 [-π,π]。φ 指弯曲平面角，不是独立的轴向扭转角，本模型不反演扭转或切向载荷。

## 最小二乘解耦

利用 sin(q+φ)=sin(q)cosφ+cos(q)sinφ，定义 a=κcosφ，b=κsinφ：

```
X1 = -Cscale*r*Ψ(s)*sin(q)
X2 = -Cscale*r*Ψ(s)*cos(q)
y = X1*a + X2*b + offset
κ = sqrt(a²+b²)
φ = atan2(b,a)
```

这与固定几何、频率、包络下的论文模型完全等价。OLS 用 NumPy `lstsq`，避免构造 XᵀX 放大条件数。拟合前对列缩放，检测秩与条件数。
Tukey IRLS 从 OLS 初始化，迭代 residual→MAD 噪声估计→权重→加权最小二乘：

```
σ = max(1.4826 median(|residual-median(residual)|), noise_floor)
u = residual/(4.685 σ)
w = (1-u²)²  if |u|<1 else 0
```

它对应论文式 (9) 的双权重损失；默认启用。`noise_floor_strain` 防止无噪声仿真中 MAD 塌陷。全局偏置默认拟合，可 `fit_offset=False` 回到无偏置的原方程。极大接触覆盖率、分布式压力或非 PCC 形变可使分解失效；IRLS 收敛不是物理模型有效的保证。

## 接触位置和大小

对 |ε_res| 使用 SciPy find_peaks，门限 50 με、显著性 20 με。额外用光纤物理距离进行非极大值抑制，默认间距 25 mm（实现选择，可配置），不依赖均匀采样。NaN 分隔的区间分别处理，不把缺测区插值成接触。区间端点的峰不检测，过密/重叠峰不能保证分开；位置取离散采样峰，精度受采样间距和噪声限制。

z=(L/Lf)s，ρ=z mod (0.017+0.009)。ρ<0.017 为 gap，其余为 disk。

```
F = max(0, (|ε_res_peak|-intercept)/slope)
```

论文表 III 单位经用户确认按 με/N、με：

| 段 | disk slope | disk intercept | gap slope | gap intercept |
|---|---:|---:|---:|---:|
| 1 | 27.914 | 35.526 | 16.845 | 42.556 |
| 2 | 25.031 | 31.820 | 14.463 | 43.853 |

配置已乘 1e-6 转为 strain。保留应变峰符号，输出力为法向载荷大小；不能从峰符号解释成力的世界坐标方向。未实现独立外力跨圈聚类，四个相邻圈的峰可能是同一接触足迹。

## 三维坐标

按 MATLAB 的弯曲螺旋扩展，主干位置为 [cosφ(1-cosκz)/κ, sinφ(1-cosκz)/κ, sinκz/κ]，截面旋转为 Rz(φ)Ry(κz)Rz(-φ)。将未变形横截面偏置 [r cos(s/λ),r sin(s/λ),0] 旋转后加到主干。

用 sinc 公式实现 κ→0 的连续极限；输出在当前段的基座坐标系。第二段输出不是整机世界坐标，应在外部施加第一段末端变换后再使用。接触造成的局部几何凹陷未纳入 PCC 定位。

## 有意处理的源代码/论文差异

1. 论文式 (7) 将光纤 s 与主干 L 混用。这里改写为 s/Lf，等价于先映射 z 后用 z/L；使 8 个包络区间覆盖整段，与 MATLAB 拟合一致。
2. MATLAB 合成使用 ceil 分区、sin(q/10)，拟合使用 floor 分区、sin(πN s/Lf)，存在相位/频率差别。Python 正向和反向共用同一 design。
3. MATLAB 用 T=.107 定义 ω=2πλ/T；Python 默认使用表 II ω=.999。复现旧 MATLAB 时可设置 `frequency_modifier=2*pi*lambda/0.107`。
4. MATLAB `opts.Robust='Bisquare'` 被注释；Python 默认启用 Tukey。未照搬 κ≤2 的优化上界，否则 0.2 m 主干弯曲超过 22.9° 无法恢复。
5. MATLAB 将 NaN 置零，Python 仅从拟合中排除，保留缺测。
6. MATLAB 用统一 c_k=1.2e-3，独立标定脚本则使用另一组数据；Python 使用论文最终表 III 的分段、分区标定。
7. 三维主干坐标采用 L/Lf 与标定分区一致；理想几何 h/(2πλ) 与 L/Lf 因表格参数舍入略有不同。
8. 论文 8×17+7×9=199 mm，与 L=200 mm 存在 1 mm 差异。分区保留式 (10) 的模运算约定，末端剩余长度仍归入 gap。AM 的 8 等分与 disk/gap 的 26 mm 周期是两个不同经验定义，没有悄悄合并。
9. 拟合 φ 与几何方位依论文及 MATLAB 共用，实际缠绕起点、正方向和 sin/cos 约定需要实验校准。配置提供两个独立相位偏置。

## 正向接触核的适用范围

`simulate` 用 A exp(-(s-sc)²/(2σs²))，A=sign*(slope*F+intercept)，只作为局部应变数据生成近似。F=0 时不添加截距峰。没有材料弹性模量、接触头尺寸、硅胶传递函数等信息，不能把这个核当作从接触力推导的连续介质解析解。反向算法直接检测残差，不强制假设真实峰为高斯。
