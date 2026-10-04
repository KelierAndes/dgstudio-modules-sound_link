"""音频联动桥接：声音采集 + 实时分析 + 核心映射与外部脉冲流推流。

数据流：sounddevice 输入流（麦克风或 WASAPI 回环）→ 音频块环形队列 →
引擎循环每 0.1s 取窗分析（与核心脉冲帧 100ms 对齐）→ 每声道得到
**响度 (0-100)** 与 **主频率 (Hz)**，作为四个映射变量喂进模块映射引擎，
同时把频率（对数映射到设备逻辑频率 10-1000）与响度（作为脉冲电平）
经 ``ctx.push_pulse_stream`` 推入核心外部脉冲流——核心据此逐帧生成
波形，不使用内置波形发生器。

映射关系全部落在核心统一的输入映射表上（配置项 ``mappings``），表为空时
按「响度×2 驱动同侧通道强度」落地默认行，安装即可用。
"""

from __future__ import annotations

import asyncio
import math
import time
import traceback
from collections import deque
from typing import Any, Callable

import numpy as np

from dglab.mapping import MappingEngine
from dglab.params import (build_dispatchers, core_alias_values, core_inputs,
                          device_state_values, input_ranges)
from dglab.state import family_of
from dglab.waves import PULSE_STREAM

__all__ = ["SoundBridge", "SoundConfig", "PARAM_DEFS",
           "rms_dbfs", "dbfs_to_level", "dominant_frequency",
           "hz_to_logical", "DEFAULT_MAPPINGS"]

# 分析与推流节奏：0.1s 一拍，一拍一帧（核心脉冲帧 = 100ms）
TICK_S = 0.1
# 分析窗口上限（样本数）：防采集堆积时窗口无界增长
MAX_WINDOW_SAMPLES = 8192
# 响度低于该值（0-100 标尺）视为静音：频率沿用上一拍、脉冲电平推 0
SILENCE_LEVEL = 1.0

# 四个映射变量（META["params"] 的唯一来源，模块页实时数据区据此展示）
PARAM_DEFS: dict[str, dict[str, str]] = {
    "left_loudness": {"label": "左响度", "desc": "左声道响度 0-100"},
    "right_loudness": {"label": "右响度", "desc": "右声道响度 0-100"},
    "left_frequency": {"label": "左频率", "desc": "左声道主频率 (Hz)"},
    "right_frequency": {"label": "右频率", "desc": "右声道主频率 (Hz)"},
}

# 映射表为空时的默认行：左/右响度直接驱动郊狼 A/B 强度（0-100 → 0-200）
DEFAULT_MAPPINGS: list[dict[str, str]] = [
    {"param": "in_strength_a", "expr": "{left_loudness} * 2"},
    {"param": "in_strength_b", "expr": "{right_loudness} * 2"},
]


class SoundConfig(dict):
    """音频模块配置：缺省值优先取模块声明（defaults 参数），DEFAULTS 为兜底。"""

    DEFAULTS = {
        "source": "microphone",
        "device": "",
        "gain": 1.0,
        "min_db": -60.0,
        "max_db": -10.0,
        "smooth": 0.5,
        "min_hz": 20.0,
        "max_hz": 2000.0,
        "mappings": [],
        "outputs": [],
    }

    def __init__(self, data: dict | None = None, defaults: dict | None = None):
        super().__init__({k: v for k, v in
                          (defaults or self.DEFAULTS).items()})
        if data:
            self.update({k: v for k, v in data.items() if v is not None})


# ---------------------------------------------------------------- 纯分析函数

def rms_dbfs(window: np.ndarray) -> float:
    """采样窗口 RMS 电平 → dBFS（空窗 / 数字静音返回 -120）。"""
    if window is None or window.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(window.astype(np.float64)))))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


def dbfs_to_level(db: float, min_db: float, max_db: float) -> float:
    """dBFS → 0-100 响度（min_db 记 0%、max_db 记 100%，线性）。"""
    span = max(1.0, float(max_db) - float(min_db))
    t = (float(db) - float(min_db)) / span
    return max(0.0, min(100.0, t * 100.0))


def dominant_frequency(window: np.ndarray, samplerate: float,
                       low_hz: float, high_hz: float) -> float:
    """Hann 窗 FFT 主峰频率（Hz，抛物线插值细化）；无有效峰返回 0。

    只在 ``[low_hz, high_hz]`` 频带内找峰，滤掉带外能量（如电源嗡声、
    高频嘶声）对主频率的干扰。
    """
    n = int(window.size) if window is not None else 0
    if n < 64 or samplerate <= 0:
        return 0.0
    x = window.astype(np.float64) * np.hanning(n)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, 1.0 / float(samplerate))
    band = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
    if band.size == 0:
        return 0.0
    if float(spec[band].max()) <= 1e-9:
        return 0.0                     # 频带内无能量（数字静音）避免峰落带边
    peak_rel = int(np.argmax(spec[band]))
    k = int(band[0]) + peak_rel
    if k <= 0 or k >= len(spec) - 1:
        return float(freqs[k])
    a, b, c = float(spec[k - 1]), float(spec[k]), float(spec[k + 1])
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    delta = max(-0.5, min(0.5, delta))
    return float((k + delta) * float(samplerate) / n)


def hz_to_logical(hz: float, low_hz: float, high_hz: float) -> int:
    """声音频率 → 设备逻辑频率 10-1000（min_hz→10、max_hz→1000，对数刻度）。"""
    lo = max(1.0, float(low_hz))
    hi = max(lo * 2.0, float(high_hz))
    hz = min(max(float(hz), lo), hi)
    t = math.log(hz / lo) / math.log(hi / lo)
    return 10 + int(round(t * 990))


# ---------------------------------------------------------------- 桥接器

class SoundBridge:
    """音频采集与分析运行时：四个映射变量 + 外部脉冲流推流。

    :param config: 模块配置（SoundConfig，reload_config 时原位更新）
    :param get_state: 引擎状态回调（``ctx.engine.get_state``）
    :param commands: 引擎命令层（``ctx.engine``，需要 wave_selection /
                     push_pulse_stream / set_strength / set_wave 等）
    :param events: 应用事件总线（可选，用于订阅 App 按键反馈）
    """

    def __init__(self, config: SoundConfig, get_state: Callable[[], Any],
                 commands: Any, events=None):
        self.config = config
        self.get_state = get_state
        self.commands = commands

        self.log: Callable[[str], None] = print
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

        # 音频采集状态（回调线程只 append 块副本，分析在引擎循环完成）
        self._stream: Any = None
        self._opened_sig: tuple | None = None   # 当前输入流对应的 (来源, 设备)
        self._blocks: deque[np.ndarray] = deque(maxlen=64)
        self._samplerate = 48000.0
        self._channels = 2
        self._last_open_error: tuple[float, str] | None = None

        # 分析状态
        self._smoothed: dict[str, float] = {}
        self._last_freq: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.last_pushes: list[tuple[int, str, int]] = []
        self.last_values: dict[str, float] = {}

        # 映射引擎：信号空间 = 音频变量 ∪ 核心输出参数实时值
        self.engine = MappingEngine(self._dispatch,
                                    device_vars=self._device_vars,
                                    ranges=input_ranges())
        self._api = self._DeviceApi(self)
        self.dispatchers = build_dispatchers(self._api, core_inputs())
        self._primed = False
        self.apply_config()

    # ---- 映射表 ---------------------------------------------------------

    def apply_config(self) -> None:
        """装载映射表；首轮只静默求值，避免启动即把设备写成 0。"""
        first = not self._primed
        if first:
            self.engine.armed = False
        self.engine.set_mappings(self._effective_rows())
        self.engine.set_outputs(self.config.get("outputs") or [])
        if first:
            self.engine.armed = True
            self._primed = True

    def _effective_rows(self) -> list[dict]:
        rows = [row for row in (self.config.get("mappings") or [])
                if isinstance(row, dict)
                and str(row.get("param") or "").strip()]
        return rows or [dict(row) for row in DEFAULT_MAPPINGS]

    def _safe_state(self):
        try:
            return self.get_state()
        except Exception:
            return None

    def _device_vars(self) -> dict[str, float]:
        """表达式可用的核心输出参数实时值 + 短名别名。"""
        vals = device_state_values(self._safe_state())
        vals.update(core_alias_values(vals))
        return vals

    def _dispatch(self, target: str, value: int) -> None:
        runner = self.dispatchers.get(target)
        if runner is None:
            return
        try:
            runner(value)
        except Exception as exc:
            self.log(f"映射派发 {target}={value} 失败: {exc!r}")

    class _DeviceApi:
        """把引擎命令层适配成核心参数派发器需要的接口。"""

        def __init__(self, bridge: "SoundBridge"):
            self._b = bridge

        @property
        def _cmd(self):
            return self._b.commands

        def resolve_slot(self, family: str = "") -> str | None:
            state = self._b._safe_state()
            if state is None:
                return None
            slots = {sid: state.slots[sid] for sid in sorted(state.slots)}
            if family:
                for sid, slot in slots.items():
                    if family_of(slot.type) == family:
                        return sid
            for sid, slot in slots.items():
                if family_of(slot.type) != "BMTR":
                    return sid
            return next(iter(slots), None)

        def wave_order(self, family: str = "") -> list[str]:
            from dglab.waves import wave_order
            return wave_order(family or "COYOTE")

        def wave_selection(self) -> dict:
            getter = getattr(self._cmd, "wave_selection", None)
            return (getter() or {}) if getter is not None else {}

        def set_strength(self, channel, value, slot_id=None):
            return self._cmd.set_strength(channel, value, slot_id=slot_id)

        def set_wave(self, channel, name, slot_id=None):
            return self._cmd.set_wave(channel, name, slot_id=slot_id)

        def zap(self, channel, seconds=1.0, slot_id=None):
            return self._cmd.zap(channel, seconds, slot_id=slot_id)

        def fire_start(self, slot_id=None):
            return self._cmd.fire_start(slot_id=slot_id)

        def fire_stop(self, slot_id=None):
            return self._cmd.fire_stop(slot_id=slot_id)

        def emergency_stop(self):
            return self._cmd.emergency_stop()

        def run(self, coro) -> None:
            self._b._spawn(coro)

    # ---- 生命周期 -------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._opened_sig = None            # 强制本循环内打开设备
        self._task = asyncio.create_task(self._tick_loop())
        self.log("音频联动已启动（每 0.1s 分析一拍：响度 / 频率 → 映射与脉冲流）")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._close_stream()
        self.log("音频联动已停止")

    def close(self) -> None:
        """模块卸载清理（无事件订阅，仅占位与后续扩展）。"""

    async def reload_config(self) -> None:
        """联动页保存设置后由宿主调用：映射表热生效，来源/设备变化重开流。

        增益 / dB 上下限 / 平滑 / 音高上下限每拍读取配置，无需特殊处理；
        输入流按「已开流签名」比对——仅来源/设备真正变化才重开。
        """
        self.apply_config()

    # ---- 音频采集 -------------------------------------------------------

    def _stream_signature(self) -> tuple:
        return (str(self.config.get("source") or "microphone"),
                str(self.config.get("device") or ""))

    def _ensure_stream(self) -> None:
        """按当前配置确保输入流可用（来源/设备变化时重开；失败退避重试）。"""
        sig = self._stream_signature()
        if self._stream is not None and sig == self._opened_sig:
            return
        if self._stream is not None:
            self._close_stream()
        try:
            self._open_stream()
            self._opened_sig = sig
            self._last_open_error = None
        except Exception as exc:
            self._stream = None
            now = time.monotonic()
            if self._last_open_error is None or now - self._last_open_error[0] > 5.0:
                self._last_open_error = (now, str(exc))
                self.log(f"音频输入流打开失败（5 秒后自动重试）: {exc}")
            self._opened_sig = None

    def _resolve_device(self):
        """按配置挑设备：loopback 用播放设备（WASAPI 回环），否则录音设备。

        ``device`` 留空用系统默认；填名称子串则匹配第一个包含它的设备。
        返回 (设备序号, 设备信息) 或 (None, None)。
        """
        import sounddevice as sd

        loopback = str(self.config.get("source")) == "loopback"
        needle = str(self.config.get("device") or "").strip().lower()
        default_in, default_out = sd.default.device
        for idx, dev in enumerate(sd.query_devices()):
            max_in = int(dev.get("max_input_channels") or 0)
            max_out = int(dev.get("max_output_channels") or 0)
            channels = max_out if loopback else max_in
            if channels <= 0:
                continue
            if not needle:
                if loopback and idx != default_out:
                    continue
                if not loopback and idx != default_in:
                    continue
            elif needle not in str(dev.get("name") or "").lower():
                continue
            return idx, dev
        return None, None

    def _open_stream(self) -> None:
        import sounddevice as sd

        idx, dev = self._resolve_device()
        if idx is None:
            raise RuntimeError("未找到匹配的音频设备（检查「声音来源/音频设备」设置）")
        loopback = str(self.config.get("source")) == "loopback"
        max_channels = int(dev.get("max_output_channels" if loopback
                                   else "max_input_channels") or 0)
        self._channels = max(1, min(2, max_channels))
        self._samplerate = float(dev.get("default_samplerate") or 48000.0)
        extra = None
        if loopback:
            try:
                extra = sd.WasapiSettings(loopback=True)
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(f"当前平台不支持 WASAPI 回环采集: {exc}") from None
        self._stream = sd.InputStream(
            samplerate=self._samplerate, channels=self._channels, device=idx,
            blocksize=0, dtype="float32", callback=self._on_audio,
            extra_settings=extra,
        )
        self._stream.start()
        self.log(f"音频输入已打开: {dev.get('name')} "
                 f"({self._samplerate:.0f} Hz × {self._channels} 声道"
                 f"{'，WASAPI 回环' if loopback else ''})")

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    def _on_audio(self, indata, frames, time_info, status) -> None:
        """sounddevice 回调（音频线程）：只做块拷贝入队，不做分析。"""
        if status:
            pass                     # overflow 等瞬时状态不处理，靠窗口截断兜底
        self._blocks.append(np.array(indata, copy=True))

    # ---- 分析与推流 -----------------------------------------------------

    async def _tick_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(TICK_S)
                try:
                    self._tick()
                except Exception:
                    self._log_error("分析节拍失败")
        except asyncio.CancelledError:
            pass

    def _take_window(self) -> np.ndarray | None:
        """取自上一拍以来的音频窗口 (samples, channels)；无数据返回 None。"""
        if not self._blocks:
            return None
        blocks = list(self._blocks)
        self._blocks.clear()
        window = np.concatenate(blocks, axis=0)
        if window.shape[0] > MAX_WINDOW_SAMPLES:
            window = window[-MAX_WINDOW_SAMPLES:]
        return window

    def _tick(self) -> None:
        if self._stream is None:
            self._ensure_stream()
        window = self._take_window()
        cfg = self.config
        gain_db = 20.0 * math.log10(max(1e-4, float(cfg.get("gain") or 1.0)))
        min_db = float(cfg.get("min_db") or -60.0)
        max_db = float(cfg.get("max_db") or -10.0)
        smooth = max(0.0, min(0.95, float(cfg.get("smooth") or 0.0)))
        low_hz = max(1.0, float(cfg.get("min_hz") or 20.0))
        high_hz = max(low_hz * 2.0, float(cfg.get("max_hz") or 2000.0))
        selection = {}
        getter = getattr(self.commands, "wave_selection", None)
        if getter is not None:
            try:
                selection = getter() or {}
            except Exception:
                selection = {}

        pushes: list[tuple[int, str, int]] = []
        for side, channel in (("left", "A"), ("right", "B")):
            column = self._channel_column(window, 0 if side == "left" else 1)
            level_raw = dbfs_to_level(rms_dbfs(column) + gain_db,
                                      min_db, max_db)
            # 静音快切（不平滑，声音一停立即归零）；起音走指数平滑
            prev = self._smoothed.get(side)
            if level_raw < SILENCE_LEVEL or prev is None:
                level = level_raw
            else:
                level = smooth * prev + (1.0 - smooth) * level_raw
            self._smoothed[side] = level

            if level < SILENCE_LEVEL:
                freq = self._last_freq.get(side, 0.0)
            else:
                freq = dominant_frequency(column, self._samplerate,
                                          low_hz, high_hz)
                if freq > 0.0:
                    self._last_freq[side] = freq
                else:
                    freq = self._last_freq.get(side, 0.0)

            self.engine.signal(f"{side}_loudness", round(level, 1))
            self.engine.signal(f"{side}_frequency", round(freq, 1))

            # 外部脉冲流：选中该通道波形时按拍推流（频率 + 响度电平）；
            # 静音时电平推 0（该帧无声但保留频率成形）
            if selection.get(channel) == PULSE_STREAM:
                logical = hz_to_logical(freq or low_hz, low_hz, high_hz)
                level_out = int(round(level))
                pushes.append((logical, channel, level_out))

        self.last_values = {name: float(self.engine.signals.get(name, 0.0) or 0.0)
                            for name in PARAM_DEFS}
        self.last_pushes = pushes
        for logical, channel, level_out in pushes:
            self._spawn(self.commands.push_pulse_stream(
                logical, channel=channel, level=level_out))

    @staticmethod
    def _channel_column(window: np.ndarray | None, index: int) -> np.ndarray:
        """取窗口某声道样本列；单声道/无数据返回安全空列或首列。"""
        if window is None or window.size == 0:
            return np.zeros(0, dtype=np.float32)
        if window.ndim < 2 or window.shape[1] <= index:
            return window.reshape(-1)
        return window[:, index]

    def _spawn(self, coro) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            coro.close()
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop)

    def _log_error(self, prefix: str) -> None:
        self.log(f"{prefix}:\n{traceback.format_exc()}")
