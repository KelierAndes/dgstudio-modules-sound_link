from __future__ import annotations

import asyncio
import math
import time
import traceback
from collections import deque
from typing import Any, Callable

import numpy as np

__all__ = ["SoundBridge", "SoundConfig", "DEVICES", "PARAM_DEFS",
           "list_microphones", "list_speakers", "migrate_settings",
           "rms_dbfs", "dbfs_to_level", "dominant_frequency", "hz_to_logical"]

TICK_S = 0.1
MAX_WINDOW_SAMPLES = 8192
SILENCE_LEVEL = 1.0

DEVICES: tuple[tuple[str, str], ...] = (("mic", "麦克风"), ("loop", "系统声音"))
SIDES: tuple[tuple[str, str], ...] = (("left", "左"), ("right", "右"))

PARAM_DEFS: dict[str, dict[str, str]] = {}
for _dev_key, _dev_label in DEVICES:
    for _side_key, _side_label in SIDES:
        PARAM_DEFS[f"{_dev_key}_{_side_key}_loudness"] = {
            "label": f"{_dev_label} · {_side_label}响度",
            "desc": f"{_dev_label}{_side_label}声道响度 0-100"}
        PARAM_DEFS[f"{_dev_key}_{_side_key}_frequency"] = {
            "label": f"{_dev_label} · {_side_label}频率",
            "desc": f"{_dev_label}{_side_label}声道主频率 (Hz)"}


REMOVED_DEVICE_KEYS = ("mic_pulse", "loop_pulse", "mic_mappings", "loop_mappings")


def migrate_settings(settings, log=None) -> bool:
    """把历史配置迁移到「纯输入」形态。

    1. 旧版「单一声音来源」（source=loopback/microphone）迁移为双设备开关：
       选 loopback 的用户升级后仍只监听系统声音（麦克风那一路关掉）。
    2. 模块不再直写设备：脉冲流推入（*_pulse）与映射表（*_mappings）配置项
       已从配置清除，改由「事件流」页面用写入卡片驱动。
    """
    changed = False
    if "source" in settings:
        source = str(settings.pop("source") or "microphone")
        loopback = source == "loopback"
        settings["mic_enabled"] = not loopback
        settings["loop_enabled"] = loopback
        if log is not None:
            log("音频联动：已把「声音来源」设置迁移为麦克风 / 系统声音双开关")
        changed = True
    dropped = False
    for key in REMOVED_DEVICE_KEYS:
        if key in settings:
            settings.pop(key)
            dropped = True
    if dropped:
        changed = True
        if log is not None:
            log("音频联动：脉冲流推入与映射表已改由「事件流」页面用写入卡片驱动，"
                "模块不再直写设备，已清除相应旧配置项")
    return changed


class SoundConfig(dict):

    DEFAULTS = {
        "mic_enabled": True,
        "microphone": "",
        "loop_enabled": False,
        "speaker": "",
        "swap_channels": False,
        "gain": 1.0,
        "min_db": -60.0,
        "max_db": -10.0,
        "smooth": 0.5,
        "min_hz": 20.0,
        "max_hz": 1000.0,
    }

    def __init__(self, data: dict | None = None, defaults: dict | None = None):
        super().__init__({k: v for k, v in
                          (defaults or self.DEFAULTS).items()})
        if data:
            self.update({k: v for k, v in data.items() if v is not None})


def _wasapi_api_index(sd) -> int | None:
    try:
        for i, api in enumerate(sd.query_hostapis()):
            if "wasapi" in str(api.get("name") or "").lower():
                return i
    except Exception:
        pass
    return None


def list_microphones() -> list[str]:
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        wasapi = _wasapi_api_index(sd)
        out: list[str] = []
        for dev in sd.query_devices():
            if int(dev.get("max_input_channels") or 0) <= 0:
                continue
            if wasapi is not None and dev.get("hostapi") != wasapi:
                continue
            name = str(dev.get("name") or "").strip()
            if name and name not in out:
                out.append(name)
        return out
    except Exception:
        return []


def list_speakers() -> list[str]:
    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        return []
    pa = None
    try:
        pa = pyaudio.PyAudio()
        out: list[str] = []
        for dev in pa.get_loopback_device_info_generator():
            name = str(dev.get("name") or "").strip()
            if name.endswith(" [Loopback]"):
                name = name[: -len(" [Loopback]")]
            if name and name not in out:
                out.append(name)
        return out
    except Exception:
        return []
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass


def rms_dbfs(window: np.ndarray) -> float:
    if window is None or window.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(window.astype(np.float64)))))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


def dbfs_to_level(db: float, min_db: float, max_db: float) -> float:
    span = max(1.0, float(max_db) - float(min_db))
    t = (float(db) - float(min_db)) / span
    return max(0.0, min(100.0, t * 100.0))


def dominant_frequency(window: np.ndarray, samplerate: float,
                       low_hz: float, high_hz: float) -> float:
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
        return 0.0
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
    # 声音频率 → 设备逻辑频率 10-1000（min_hz→10、max_hz→1000，对数刻度）。
    # 模块不再用它直推核心：这段换算现由「事件流」里的范围映射 / 公式卡对
    # {…_frequency} 变量完成。保留导出供测试与 README 引用。
    lo = max(1.0, float(low_hz))
    hi = max(lo * 2.0, float(high_hz))
    hz = min(max(float(hz), lo), hi)
    t = math.log(hz / lo) / math.log(hi / lo)
    return 10 + int(round(t * 990))


class _SignalHub:
    """两路采集各持一个信号字典；对核心只暴露合并后的只读信号视图。

    核心 `plugins._mapping_engine` 与 `flow_host.module_signals` 只认「一个
    engine」对象，需要 `.signals`（事件流变量表读数）与 `.errors`；`.pump()`
    仅为兼容宿主 set_temp/apply_logic_tables 的调用点而保留的空操作。本模块
    不再做任何设备派发，频率→设备频率的换算改在事件流里用范围映射卡完成。
    """

    def __init__(self, sources: dict[str, "_Source"]):
        self._sources = sources

    @property
    def signals(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for source in self._sources.values():
            out.update(source.signals)
        return out

    @property
    def errors(self) -> dict[str, str]:
        return {}

    def pump(self) -> None:
        return None


class _Source:
    """一路声音采集：设备句柄、采样块、平滑状态与本路信号字典。"""

    def __init__(self, key: str, label: str, bridge: "SoundBridge"):
        self.key = key
        self.label = label
        self.bridge = bridge
        self.blocks: deque[np.ndarray] = deque(maxlen=64)
        self.stream: Any = None
        self.pa: Any = None
        self.pa_stream: Any = None
        self.opened_sig: tuple | None = None
        self.open_error: tuple[float, str] | None = None
        self.samplerate = 48000.0
        self.channels = 2
        self.smoothed: dict[str, float] = {}
        self.last_freq: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.signals: dict[str, float] = {}

    @property
    def config(self) -> dict:
        return self.bridge.config

    def is_enabled(self) -> bool:
        return bool(self.config.get(f"{self.key}_enabled"))

    def device_name(self) -> str:
        key = "speaker" if self.key == "loop" else "microphone"
        return str(self.config.get(key) or "").strip()

    def signature(self) -> tuple:
        return (self.is_enabled(), self.device_name())

    def ensure(self) -> bool:
        sig = self.signature()
        if self.stream is not None or self.pa_stream is not None:
            if sig == self.opened_sig:
                return True
            self.close()
        if not sig[0]:
            self.opened_sig = None
            return False
        try:
            if self.key == "loop":
                self._open_loopback()
            else:
                self._open_microphone()
            self.opened_sig = sig
            self.open_error = None
            return True
        except Exception as exc:
            self.close()
            now = time.monotonic()
            if self.open_error is None or now - self.open_error[0] > 5.0:
                self.open_error = (now, str(exc))
                self.bridge.log(f"{self.label}采集打开失败（5 秒后自动重试）: {exc}")
            self.opened_sig = None
            return False

    def close(self) -> None:
        pa_stream, self.pa_stream = self.pa_stream, None
        if pa_stream is not None:
            try:
                pa_stream.stop_stream()
                pa_stream.close()
            except Exception:
                pass
        pa, self.pa = self.pa, None
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        stream, self.stream = self.stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    def take_window(self) -> np.ndarray | None:
        if not self.blocks:
            return None
        blocks = list(self.blocks)
        self.blocks.clear()
        window = np.concatenate(blocks, axis=0)
        if window.shape[0] > MAX_WINDOW_SAMPLES:
            window = window[-MAX_WINDOW_SAMPLES:]
        return window

    def column(self, window: np.ndarray | None, index: int) -> np.ndarray:
        if window is None or window.size == 0:
            return np.zeros(0, dtype=np.float32)
        if window.ndim < 2 or window.shape[1] <= index:
            return window.reshape(-1)
        return window[:, index]

    def on_audio(self, indata, frames, time_info, status) -> None:
        self.blocks.append(np.array(indata, copy=True))

    def pa_callback(self, indata, frame_count, time_info, status):
        try:
            arr = np.frombuffer(indata, dtype=np.float32)
            arr = arr.reshape(-1, self.channels) if self.channels > 1 \
                else arr.reshape(-1, 1)
            self.blocks.append(arr.copy())
        except Exception:
            pass
        import pyaudiowpatch as pyaudio
        return None, pyaudio.paContinue

    def _log_bound(self, device_name: str) -> None:
        swap = bool(self.config.get("swap_channels"))
        lcol, rcol = (1, 0) if swap else (0, 1)
        self.bridge.log(
            f"{self.label}采集已打开: {device_name} "
            f"({self.samplerate:.0f} Hz × {self.channels} 声道)\n"
            f"声道绑定: 采集声道{lcol} → {self.key}_left_*，声道{rcol} → "
            f"{self.key}_right_*（频率/响度只登记为只读变量，设备动作请在"
            f"「事件流」里用写入卡片驱动；左右接反时开启「左右声道交换」对调）")

    def _open_microphone(self) -> None:
        import sounddevice as sd

        idx, dev = self._resolve_microphone(sd)
        if idx is None:
            raise RuntimeError("未找到匹配的录音设备（检查「麦克风设备」设置）")
        max_in = int(dev.get("max_input_channels") or 0)
        self.channels = max(1, min(2, max_in))
        self.samplerate = float(dev.get("default_samplerate") or 48000.0)
        self.stream = sd.InputStream(
            samplerate=self.samplerate, channels=self.channels, device=idx,
            blocksize=0, dtype="float32", callback=self.on_audio,
        )
        self.stream.start()
        self._log_bound(str(dev.get("name") or idx))

    def _resolve_microphone(self, sd):
        name = self.device_name()
        devices = sd.query_devices()
        wasapi = _wasapi_api_index(sd)
        default_idx = None
        if wasapi is not None:
            hostapis = sd.query_hostapis()
            default_idx = int(hostapis[wasapi].get("default_input_device") or -1)

        candidates: list[tuple[bool, int, dict]] = []
        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels") or 0) <= 0:
                continue
            candidates.append((dev.get("hostapi") == wasapi, idx, dev))
        if not candidates:
            return None, None
        candidates.sort(key=lambda c: not c[0])

        if not name:
            if default_idx is not None and default_idx >= 0:
                return default_idx, devices[default_idx]
            return candidates[0][1], candidates[0][2]

        for match in (lambda n: n == name, lambda n: name in n):
            for _preferred, idx, dev in candidates:
                if match(str(dev.get("name") or "")):
                    return idx, dev
        raise RuntimeError(f"未找到匹配「{name}」的录音设备")

    def _resolve_loopback(self, pa, name: str):
        import pyaudiowpatch as pyaudio

        loopbacks = list(pa.get_loopback_device_info_generator())
        if not loopbacks:
            raise RuntimeError("未枚举到任何可回环采集的播放设备")
        if name:
            for dev in loopbacks:
                if name in str(dev.get("name") or ""):
                    return dev
            raise RuntimeError(f"未找到包含「{name}」的回环播放设备")
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = pa.get_device_info_by_index(
            int(wasapi["defaultOutputDevice"]))
        if default_speakers.get("isLoopbackDevice"):
            return default_speakers
        default_name = str(default_speakers.get("name") or "")
        for dev in loopbacks:
            if default_name and default_name in str(dev.get("name") or ""):
                return dev
        return loopbacks[0]

    def _open_loopback(self) -> None:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            target = self._resolve_loopback(pa, self.device_name())
            channels = max(1, int(target.get("maxInputChannels") or 2))
            rate = int(target.get("defaultSampleRate") or 48000)
            self.pa_stream = pa.open(
                format=pyaudio.paFloat32, channels=channels, rate=rate,
                input=True, input_device_index=int(target["index"]),
                frames_per_buffer=int(rate * 0.04),
                stream_callback=self.pa_callback,
            )
            self.pa_stream.start_stream()
            self.pa = pa
            self.channels = channels
            self.samplerate = float(rate)
            self._log_bound(str(target.get("name") or "系统默认播放设备"))
        except Exception:
            try:
                pa.terminate()
            except Exception:
                pass
            raise


class SoundBridge:

    def __init__(self, config: SoundConfig, get_state: Callable[[], Any] = None,
                 commands: Any = None, events=None):
        self.config = config
        self.get_state = get_state
        self.commands = commands
        self.events = events

        self.log: Callable[[str], None] = print
        self._running = False
        self.last_values: dict[str, float] = {}

        self.sources = {key: _Source(key, label, self)
                        for key, label in DEVICES}
        self.engine = _SignalHub(self.sources)

    # ---------------------------------------------------------------- 生命周期
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for source in self.sources.values():
            source.ensure()
        self._task = asyncio.create_task(self._tick_loop())
        self.log("音频联动已启动（每 0.1s 一拍：双路采集各自分析响度 / 频率，"
                 "结果登记为八个只读变量；设备动作由「事件流」写入卡驱动）")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        for source in self.sources.values():
            source.close()
        self.log("音频联动已停止")

    def close(self) -> None:
        pass

    def _log_error(self, prefix: str) -> None:
        self.log(f"{prefix}:\n{traceback.format_exc()}")

    # ---------------------------------------------------------------- 分析节拍
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

    def _tick(self) -> None:
        cfg = self.config
        gain_db = 20.0 * math.log10(max(1e-4, float(cfg.get("gain") or 1.0)))
        min_db = float(cfg.get("min_db") or -60.0)
        max_db = float(cfg.get("max_db") or -10.0)
        smooth = max(0.0, min(0.95, float(cfg.get("smooth") or 0.0)))
        low_hz = max(1.0, float(cfg.get("min_hz") or 20.0))
        high_hz = max(low_hz * 2.0, min(1000.0, float(cfg.get("max_hz") or 1000.0)))
        swap = bool(cfg.get("swap_channels"))

        for source in self.sources.values():
            if not source.ensure():
                for side, _label in SIDES:
                    source.signals[f"{source.key}_{side}_loudness"] = 0.0
                    source.signals[f"{source.key}_{side}_frequency"] = 0.0
                continue
            window = source.take_window()
            for side, audio_col in (("left", 1 if swap else 0),
                                    ("right", 0 if swap else 1)):
                level, freq = self._measure(
                    source, side, window, audio_col, gain_db, min_db, max_db,
                    smooth, low_hz, high_hz)
                source.signals[f"{source.key}_{side}_loudness"] = round(level, 1)
                source.signals[f"{source.key}_{side}_frequency"] = round(freq, 1)

        self.last_values = dict(self.engine.signals)

    def _measure(self, source: _Source, side: str, window, audio_col: int,
                 gain_db: float, min_db: float, max_db: float, smooth: float,
                 low_hz: float, high_hz: float) -> tuple[float, float]:
        column = source.column(window, audio_col)
        level_raw = dbfs_to_level(rms_dbfs(column) + gain_db, min_db, max_db)
        prev = source.smoothed.get(side)
        if level_raw < SILENCE_LEVEL or prev is None:
            level = level_raw
        else:
            level = smooth * prev + (1.0 - smooth) * level_raw
        source.smoothed[side] = level

        if level < SILENCE_LEVEL:
            return level, source.last_freq.get(side, 0.0)
        freq = dominant_frequency(column, source.samplerate, low_hz, high_hz)
        if freq > 0.0:
            source.last_freq[side] = freq
            return level, freq
        return level, source.last_freq.get(side, 0.0)
