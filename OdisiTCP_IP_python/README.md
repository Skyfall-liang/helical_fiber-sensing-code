# Luna ODiSI Python 实时数据接收

参考相邻 OdisiTCP_IP 中的 TcpipClientInit.m、JSONDataStruct.m、ODISIReceiver.m 和 main_ODISI.m。核心功能只需 Python 3.10+ 标准库；可选实时绘图需要 Matplotlib。既有 MATLAB 文件不修改。

## 连接实机

在此文件夹中打开终端：

```powershell
python receive.py --host 192.168.1.134 --port 50000 --duration 10
# 持续采集，可选实时绘图，Ctrl+C 或关闭图窗结束
python receive.py --host 192.168.1.134 --plot
# 只保留通道 1 和 4 的 latest/history/快照（原始消息日志仍保存全部消息）
python receive.py --host 10.1.1.195 --channels 1 4 --plot
```

请替换为实际控制器 IP。参考 main_ODISI.m 使用 192.168.1.134，顶层 main.m 使用 10.1.1.195，旧 V2 使用 192.168.1.129，不能据此认定当前在线地址。需在 Luna 软件中启用相应 TCP 数据输出并确保端口可访问。

默认发送 `START\n`，退出发送 `remote:stop\n`（跟随当前 ODISIReceiver.stop）。旧脚本使用 STOP，如果仪器需要旧命令：

```powershell
python receive.py --host 192.168.1.134 --stop-command STOP
# 不发送任何启动/停止命令，只接收已开启的数据流
python receive.py --host 192.168.1.134 --start-command "" --stop-command ""
```

这些命令来自现有 MATLAB 的行为，尚未通过实际仪器验证。程序不会自动重连或重新启动测量，断线后保存当前数据并退出，需显式重运行。

## 输出

每次运行新建 `recordings/年月日_时分秒_微秒/`，可用 `--output 新目录` 指定，拒绝覆盖已有目录。

- `messages.jsonl`：每条已解析 JSON 的完整字段、接收 UTC 时间与连接编号，包括 metadata/tare/measurement；无效非 JSON 行用统计计数记录，不保存原始字节。
- `stats.json`：有效测量数、错误/丢弃行、字节数等。
- `latest/channel_N.csv`：退出时各通道最新测量，默认 `index,strain`。
- `latest/snapshot.json`：各通道独立时间戳、点数、元数据和坐标假设。

应变数值保持原始值，null/NaN/Inf 转为缺失值，不填零。绘图标签默认微应变，沿用 MATLAB；实际单位请核实仪器配置与 metadata，`--unit-label` 只改变显示文字。JSON 日志中的非有限数以 null 保存，CSV 中为 nan。

`--duration 0` 表示持续接收。核心循环没有无界历史数组，默认每通道只保留最近 10 条，完整消息顺序写盘。消息率为通道测量消息/秒，不冒充多通道同步帧率。磁盘和回调应能跟上采集速率；该同步实现未承诺达到设备最高速率。

## 光纤坐标与后续模型对接

设备测点坐标字段在参考代码里没有统一格式，因此默认导出索引，不把总长度猜成坐标。只有确认已裁出敏感段且均匀采样后，才能指定：

```powershell
python receive.py --host 192.168.1.134 --duration 10 --start-index 1099 --fiber-length 4.277
```

`start-index` 为 Python 从 0 开始索引，1099 对应 MATLAB 1100；它只影响退出时的快照，不改变原始日志/回调数据。长度指裁剪后所有点所覆盖的长度，产生 `s_m,strain`。若仍含尾纤或非均匀坐标，应另外依据仪器真实坐标裁剪，不能使用这个均匀假设。输出微应变时，接入此前网络需 `--unit microstrain`；内部理论 API 的无量纲应变需乘 1e-6。

## Python 回调

```python
from odisi_receiver import ODISIReceiver

def process(m):
    # m.data 为一个通道的一条空间应变分布 tuple，可 np.asarray(m.data)
    print(m.channel, len(m.data), m.received_at)

with ODISIReceiver('192.168.1.134', 50000, channels=[1, 2]) as receiver:
    receiver.on_measurement = process
    while True:
        receiver.poll()
```

metadata、tare 和未知类型不会触发重复测量回调；tare 仅保留，不自动再减一次基线。`latest[channel]`/`history[channel]` 保存测量。无需等待 metadata 才能接收 measurement，避免固件只发送一次元数据而永久无法工作。metadata 原始内容仍可用于外部标定。

没有协议依据可保证不同通道相同时间采样，所以不自动拼接“同步帧”。每条 measurement 保留设备 timestamp（若存在），后续若需同步，应先核对设备共同序号/时间戳语义。

## 离线测试

终端 1：
```powershell
python mock_server.py
```
终端 2：
```powershell
python receive.py --host 127.0.0.1 --port 50001 --duration 5 --plot
```

自动测试：
```powershell
python -m unittest discover -s tests -v
```

测试覆盖分包/粘包、UTF-8、前缀、坏 JSON、超长行恢复、字段别名、NaN、变长通道、回调次数、快照、真实本机 TCP 通信和截断断线。只验证软件与模拟协议，不代表已连上 Luna 实机。

## VS Code

用 VS Code 打开此 Python 子文件夹，安装 Python/Python Debugger 扩展，选择 Python 3.10+。`.vscode/launch.json` 提供实机、模拟服务器和模拟客户端入口。实机入口中的 IP 按仪器修改。绘图依赖安装：`python -m pip install -r requirements.txt`。

## 协议边界

兼容每行一个 JSON 对象，以 LF/CRLF 结束，可有 `data:` 前缀；messageType、message type、message_type 统一识别，完整原字段在日志保留。测量数据须为一维数值数组。若实际固件输出二进制、跨多行 JSON 或不同结构，应按真实报文调整；不会用字符串替换强行“修复”损坏数据。

不清空连接后的输入缓冲，避免丢失初始 metadata。默认最大单行 8 MiB，超长行丢弃至下一个换行后恢复。回调异常向上传播，调用方可定位算法错误，连接仍通过 with/finally 清理。
