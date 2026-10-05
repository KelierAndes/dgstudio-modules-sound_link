# dgstudio-modules-sound_link · 音频联动模块

DGStudio 的音频联动模块：采集**麦克风**或**系统正在播放的声音**，每 0.1 秒
分析一拍，实时维护六个映射变量（**左响度 / 右响度 / 左频率 / 右频率 /
左推流 / 右推流**）。脉冲流走**数值推入**：默认事件流卡片（每 100ms 周期）
把左右推流值推入核心参数 `in_pulse_a/b`——核心对脉冲流参数每拍生成一帧
100ms 脉冲，通道波形选「外部脉冲流 (PULSE_STREAM)」即成流（输出频率跟随
声音音高），不使用内置波形发生器。

- 模块开发文档（ModuleContext / 事件流 / 打包约定）见总仓库
  [dgstudio-modules-market/EXTENSIONS.md](https://github.com/KelierAndes/dgstudio-modules-market)。
- 本仓库与 DGStudio 核心仓库（`DG-LAB-X-VRChat-OSC-development`）放同一
  父目录即可直接跑单测（或设 `DGSTUDIO_CORE`）。

## 使用

1. DGStudio「模块」页 → 在线列表找到「音频联动」→ 安装并启动（依赖自动补装）。
2. 控制页把 A/B 通道波形切到「**外部脉冲流 (PULSE_STREAM)**」。
3. 联动页「音频联动」卡片按需调整事件流与音频设置；卡片顶部的「实时数据」
   会显示六个变量的实时值。

首次运行自动播种一张默认事件卡（联动页事件流可自行调整）：

> **音频脉冲流推入**（周期 100ms）：`left_pulse` → `in_pulse_a`、
> `right_pulse` → `in_pulse_b`——核心把每次推入转成一帧脉冲（0=静音帧，
> 10-1000=脉冲频率，电平 100）。

六个映射变量：

| 变量 | 说明 |
|---|---|
| `{left_loudness}` / `{right_loudness}` | 左/右声道响度 0-100（可用于事件流驱动强度等） |
| `{left_frequency}` / `{right_frequency}` | 左/右声道主频率 (Hz) |
| `{left_pulse}` / `{right_pulse}` | 左/右推流值：静音=0，否则=主频对数映射到设备逻辑频率 10-1000 |

## 音频设备绑定（v0.2.0）

- **下拉选择**：联动页设置区的「麦克风设备」「播放设备（回环）」自动
  **扫描系统音频设备生成下拉选择框**（麦克风=WASAPI 录音端点；播放设备
  =可回环采集的渲染端点），留空用系统默认。
- **声道绑定**：采集声道 0（物理左）→ `left_*` 变量、声道 1（物理右）→
  `right_*` 变量；设备侧绑定（哪个变量推入 A/B 通道）由事件流卡片配置。
  若现场感觉左右接反，开启「左右声道交换」把两个采集声道对调即可。
- **系统声音回环实现**：sounddevice 自带的 PortAudio 不暴露回环端点
  （对渲染设备开输入流/全双工都会报 Invalid number of channels），回环
  采集改用 **pyaudiowpatch**（WASAPI loopback 补丁版 PyAudio），自动定位
  系统默认播放设备的 `[Loopback]` 端点。
- Windows 下同一设备会以 MME / DirectSound / WASAPI / WDM-KS 等多个宿主
  API 条目重复出现，模块一律**只绑定 WASAPI 条目**（原生混音格式、回环
  唯一可用），旧版「名称包含匹配」配置仍兼容（在 WASAPI 条目内匹配）。

## 联动原理

```
麦克风 (sounddevice) / 系统声音 (pyaudiowpatch WASAPI 回环)
        │  每 0.1s 取窗分析（Hann 窗 FFT 主频 + RMS→dBFS 响度）
        ▼
六个映射变量（响度 / 频率 / 推流值 × 左右）──► 模块映射引擎信号空间
        │
        └─ 事件流周期卡（100ms，宿主 50ms 节拍驱动）
             left_pulse → in_pulse_a、right_pulse → in_pulse_b（核心输入参数）
                  └─► 核心派发器（0.1s 节流，0=静音帧）
                        └─► 引擎 push_pulse_stream → 以最新帧实时刷新
                              「外部脉冲流」波形（输出频率即时跟随声音）
```

- **频率→设备频率**：`[min_hz, max_hz]` 声音频带对数映射到设备逻辑频率
  10-1000（低音→低频脉冲，高音→高频脉冲）。
- **响度→脉冲电平**：每帧电平 0-100 跟随响度，静音帧电平为 0（无声）；
  设备总强度仍由核心强度控制（映射表 / 手动 / 按键），两层独立。
- **连接方式**：蓝牙直连与 V4 Socket 实时逐帧成流（推荐）；V3 Socket
  协议只能整段替换波形，为尽力而为（每秒重发最近 2 秒窗口）。

## 配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| 声音来源 | microphone | `microphone` 麦克风；`loopback` 系统声音（Windows WASAPI 回环） |
| 麦克风设备 | （空） | 下拉选择（自动扫描），留空用系统默认录音设备 |
| 播放设备（回环） | （空） | 下拉选择（自动扫描），loopback 来源时生效，留空用系统默认播放设备 |
| 左右声道交换 | 关 | 开启后采集声道0→右变量、声道1→左变量（现场左右接反时对调） |
| 响度增益 | 1.0 | dB 映射前乘系数，声音偏小可调大 |
| 响度 0% / 100% (dBFS) | -60 / -10 | RMS 电平线性映射到响度 0-100 |
| 响度平滑 | 0.5 | 起音指数平滑；静音立即归零（快切） |
| 音高下限 / 上限 (Hz) | 20 / 2000 | 检测频带，对数映射到设备频率 10-1000 |

> 模块为**纯输入设计**（音频数据 → 设备）：设备动作全部由联动页的
> 事件流与临时变量配置（宿主统一数据面），模块自身只喂信号。
> 响度、音高参数在联动页「保存设置」后热生效；来源/设备按签名变化
> 自动重开输入流。

## 开发

```
dgstudio-modules-sound_link/
├── modules/sound_link/plugin.py    META + 模块类（生命周期 / 动态设备下拉 / reload_config）
├── modules/sound_link/bridge.py    采集分析（麦克风 + WASAPI 回环）+ 映射引擎 + 脉冲流推流
├── modules/sound_link/requirements.txt
├── modules/sound_link/wheels/      打包运行时离线依赖（cp312 win_amd64）
└── tests/                          单测（需核心仓库同级或 DGSTUDIO_CORE）
```

```
python -m unittest discover -s tests        # 模块单测
python _tools/build_market.py --local ..    # 总仓库：本地聚合市场清单
```

依赖：`numpy`（FFT/响度）、`sounddevice`（麦克风采集）、
`pyaudiowpatch`（Windows 系统声音 WASAPI 回环采集）。

## License

MIT
