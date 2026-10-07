
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

__all__ = ["SoundBridge", "SoundConfig", "PARAM_DEFS", "DEFAULT_EVENT_CARDS",
           "list_microphones", "list_speakers",
           "rms_dbfs", "dbfs_to_level", "dominant_frequency",
           "hz_to_logical"]

TICK_S = 0.1
MAX_WINDOW_SAMPLES = 8192
SILENCE_LEVEL = 1.0

PARAM_DEFS: dict[str, dict[str, str]] = {
    "left_loudness": {"label": "左响度", "desc": "左声道响度 0-100"},
    "right_loudness": {"label": "右响度", "desc": "右声道响度 0-100"},
    "left_frequency": {"label": "左频率", "desc": "左声道主频率 (Hz)"},
    "right_frequency": {"label": "右频率", "desc": "右声道主频率 (Hz)"},
    "left_pulse": {"label": "左推流", "desc": "左声道推流值（静音=0，否则=主频）"},
    "right_pulse": {"label": "右推流", "desc": "右声道推流值（静音=0，否则=主频）"},
}

DEFAULT_EVENT_CARDS: list[dict] = [
    {"name": "音频脉冲流推入", "trigger": "period", "arg": 100,
     "actions": [
         {"dir": "in", "param": "in_pulse_a", "var": "left_pulse"},
         {"dir": "in", "param": "in_pulse_b", "var": "right_pulse"},
     ]},
]


class SoundConfig(dict):

    DEFAULTS = {
        "source": "microphone",
        "microphone": "",
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
    lo = max(1.0, float(low_hz))
    hi = max(lo * 2.0, float(high_hz))
    hz = min(max(float(hz), lo), hi)
    t = math.log(hz / lo) / math.log(hi / lo)
    return 10 + int(round(t * 990))


class SoundBridge:

    def __init__(self, config: SoundConfig, get_state: Callable[[], Any],
                 commands: Any, events=None):
        self.config = config
        self.get_state = get_state
        self.commands = commands

        self.log: Callable[[str], None] = print
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

        self._stream: Any = None
        self._pa: Any = None
        self._pa_stream: Any = None
        self._opened_sig: tuple | None = None
        self._blocks: deque[np.ndarray] = deque(maxlen=64)
        self._samplerate = 48000.0
        self._channels = 2
        self._last_open_error: tuple[float, str] | None = None

        self._smoothed: dict[str, float] = {}
        self._last_freq: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.last_values: dict[str, float] = {}

        self.engine = MappingEngine(self._dispatch,
                                    device_vars=self._device_vars,
                                    ranges=input_ranges())
        self._api = self._DeviceApi(self)
        self.dispatchers = build_dispatchers(self._api, core_inputs())

    def _safe_state(self):
        try:
            return self.get_state()
        except Exception:
            return None

    def _device_vars(self) -> dict[str, float]:
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
                return None
            for sid, slot in slots.items():
                if family_of(slot.type) != "BMTR":
                    return sid
            return next(iter(slots), None)

        def slot_family(self, sid: str) -> str:
            state = self._b._safe_state()
            if state is None or not sid:
                return ""
            slot = state.slots.get(sid)
            return family_of(slot.type) if slot is not None else ""

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

        def push_pulse(self, channel, value, level=100, slot_id=None):
            return self._cmd.push_pulse_stream(value, channel=channel,
                                               level=level, slot_id=slot_id)

        def pulse_level(self, channel: str) -> int:
            side = "left" if channel == "A" else "right"
            level = self._b._smoothed.get(side, 0.0)
            return max(0, min(100, int(round(level))))

        def zap(self, channel, seconds=1.0, slot_id=None):
            return self._cmd.zap(channel, seconds, slot_id=slot_id)

        def fire_start(self, slot_id=None, channel=None):
            return self._cmd.fire_start(slot_id=slot_id, channel=channel)

        def fire_stop(self, slot_id=None, channel=None):
            return self._cmd.fire_stop(slot_id=slot_id, channel=channel)

        def emergency_stop(self):
            return self._cmd.emergency_stop()

        def run(self, coro) -> None:
            self._b._spawn(coro)


    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._opened_sig = None
        self._task = asyncio.create_task(self._tick_loop())
        self.log("音频联动已启动（每 0.1s 分析一拍：响度 / 频率 / 推流值 → "
                 "信号空间；脉冲流由事件流周期卡推入 in_pulse_*）")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._close_stream()
        self.log("音频联动已停止")

    def close(self) -> None:
        pass


    def _stream_signature(self) -> tuple:
        return (str(self.config.get("source") or "microphone"),
                str(self.config.get("microphone") or ""),
                str(self.config.get("speaker") or ""),
                str(self.config.get("device") or ""))

    def _ensure_stream(self) -> None:
        sig = self._stream_signature()
        if self._stream is not None or self._pa_stream is not None:
            if sig == self._opened_sig:
                return
            self._close_stream()
        try:
            if str(self.config.get("source")) == "loopback":
                self._open_loopback()
            else:
                self._open_microphone()
            self._opened_sig = sig
            self._last_open_error = None
        except Exception as exc:
            self._close_stream()
            now = time.monotonic()
            if self._last_open_error is None or now - self._last_open_error[0] > 5.0:
                self._last_open_error = (now, str(exc))
                self.log(f"音频输入流打开失败（5 秒后自动重试）: {exc}")
            self._opened_sig = None

    def _bound_name(self) -> str:
        source = str(self.config.get("source") or "microphone")
        key = "speaker" if source == "loopback" else "microphone"
        return (str(self.config.get(key) or "").strip()
                or str(self.config.get("device") or "").strip())

    def _open_microphone(self) -> None:
        import sounddevice as sd

        idx, dev = self._resolve_microphone(sd)
        if idx is None:
            raise RuntimeError("未找到匹配的录音设备（检查「麦克风设备」设置）")
        max_in = int(dev.get("max_input_channels") or 0)
        self._channels = max(1, min(2, max_in))
        self._samplerate = float(dev.get("default_samplerate") or 48000.0)
        self._stream = sd.InputStream(
            samplerate=self._samplerate, channels=self._channels, device=idx,
            blocksize=0, dtype="float32", callback=self._on_audio,
        )
        self._stream.start()
        self._log_bound(str(dev.get("name") or idx))

    def _resolve_microphone(self, sd):
        name = self._bound_name()
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

    def _resolve_loopback_device(self, pa, name: str):
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
            target = self._resolve_loopback_device(pa, self._bound_name())
            channels = max(1, int(target.get("maxInputChannels") or 2))
            rate = int(target.get("defaultSampleRate") or 48000)
            self._pa_stream = pa.open(
                format=pyaudio.paFloat32, channels=channels, rate=rate,
                input=True, input_device_index=int(target["index"]),
                frames_per_buffer=int(rate * 0.04),
                stream_callback=self._pa_callback,
            )
            self._pa_stream.start_stream()
            self._pa = pa
            self._channels = channels
            self._samplerate = float(rate)
            self._log_bound(str(target.get("name") or "系统默认播放设备"))
        except Exception:
            try:
                pa.terminate()
            except Exception:
                pass
            raise

    def _log_bound(self, device_name: str) -> None:
        swap = bool(self.config.get("swap_channels"))
        source = str(self.config.get("source") or "microphone")
        lcol, rcol = (1, 0) if swap else (0, 1)
        self.log(
            f"音频输入已打开: {device_name} "
            f"({self._samplerate:.0f} Hz × {self._channels} 声道，"
            f"{'系统声音回环' if source == 'loopback' else '麦克风'}）\n"
            f"声道绑定: 采集声道{lcol} → 左变量 left_*，声道{rcol} → 右变量 "
            f"right_*；默认事件流把左右推流值推入设备 A/B 通道脉冲流"
            f"（左右接反时开启「左右声道交换」对调）")

    def _close_stream(self) -> None:
        pa_stream, self._pa_stream = self._pa_stream, None
        if pa_stream is not None:
            try:
                pa_stream.stop_stream()
                pa_stream.close()
            except Exception:
                pass
        pa, self._pa = self._pa, None
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            pass
        self._blocks.append(np.array(indata, copy=True))

    def _pa_callback(self, indata, frame_count, time_info, status):
        try:
            arr = np.frombuffer(indata, dtype=np.float32)
            arr = arr.reshape(-1, self._channels) if self._channels > 1 \
                else arr.reshape(-1, 1)
            self._blocks.append(arr.copy())
        except Exception:
            pass
        import pyaudiowpatch as pyaudio
        return None, pyaudio.paContinue


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
        if not self._blocks:
            return None
        blocks = list(self._blocks)
        self._blocks.clear()
        window = np.concatenate(blocks, axis=0)
        if window.shape[0] > MAX_WINDOW_SAMPLES:
            window = window[-MAX_WINDOW_SAMPLES:]
        return window

    def _tick(self) -> None:
        if self._stream is None and self._pa_stream is None:
            self._ensure_stream()
        window = self._take_window()
        cfg = self.config
        gain_db = 20.0 * math.log10(max(1e-4, float(cfg.get("gain") or 1.0)))
        min_db = float(cfg.get("min_db") or -60.0)
        max_db = float(cfg.get("max_db") or -10.0)
        smooth = max(0.0, min(0.95, float(cfg.get("smooth") or 0.0)))
        low_hz = max(1.0, float(cfg.get("min_hz") or 20.0))
        high_hz = max(low_hz * 2.0, min(1000.0, float(cfg.get("max_hz") or 1000.0)))

        swap = bool(self.config.get("swap_channels"))
        for side, audio_col in (("left", 1 if swap else 0),
                                ("right", 0 if swap else 1)):
            column = self._channel_column(window, audio_col)
            level_raw = dbfs_to_level(rms_dbfs(column) + gain_db,
                                      min_db, max_db)
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
            pulse = hz_to_logical(freq, low_hz, high_hz) \
                if level >= SILENCE_LEVEL and freq > 0.0 else 0
            self.engine.signal(f"{side}_pulse", pulse)

        self.last_values = {name: float(self.engine.signals.get(name, 0.0) or 0.0)
                            for name in PARAM_DEFS}

    @staticmethod
    def _channel_column(window: np.ndarray | None, index: int) -> np.ndarray:
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
