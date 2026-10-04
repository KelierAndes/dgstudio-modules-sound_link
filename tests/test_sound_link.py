"""音频联动模块单测。

覆盖：四个映射变量的分析纯函数（响度 dB 映射 / FFT 主频 / 频率→设备
逻辑频率对数映射）、桥接器一拍分析 + 外部脉冲流推流、映射表默认行与
热更新、插件 META / link_params 契约。不依赖音频硬件（sounddevice 仅在
真实开流时导入，测试用哨兵流对象绕过）。

运行（模块仓库根目录）::

    python -m unittest discover -s tests
"""
from __future__ import annotations

import ast
import math
import os
import unittest

import numpy as np

import _bootstrap  # noqa: F401  定位核心仓库并挂 sys.path

from dglab.state import EngineState, Slot
from dglab.waves import PULSE_STREAM

from modules.sound_link.bridge import (DEFAULT_MAPPINGS, PARAM_DEFS,
                                       SoundBridge, SoundConfig, dbfs_to_level,
                                       dominant_frequency, hz_to_logical,
                                       rms_dbfs)
from modules.sound_link.plugin import META, SoundLinkModule

SR = 48000.0
BLOCK = 4800            # 100ms @48k，与核心脉冲帧对齐


def _sine(freq: float, seconds: float = 0.1, amp: float = 0.5,
          channels: int = 2, per_channel: dict[int, float] | None = None
          ) -> np.ndarray:
    """立体声正弦测试块：默认全通道同频，per_channel 可逐通道指定频率。"""
    t = np.arange(int(SR * seconds)) / SR
    cols = []
    for ch in range(channels):
        f = (per_channel or {}).get(ch, freq)
        cols.append(amp * np.sin(2 * math.pi * f * t))
    return np.stack(cols, axis=1).astype(np.float32)


class FakeCommands:
    """引擎命令层假件：记录派发与推流（不触碰真实设备）。"""

    def __init__(self):
        self.state = EngineState(backend="ble")
        self.state.slots["s1"] = Slot(slot_id="s1", name="t", type="COYOTE_030")
        self.selection = {"A": PULSE_STREAM, "B": PULSE_STREAM}
        self.pushed: list[tuple[int, str, int]] = []
        self.strength_calls: list[tuple[str, int]] = []

    def get_state(self):
        return self.state

    def wave_selection(self):
        return dict(self.selection)

    async def push_pulse_stream(self, frequency, channel="A", level=100,
                                slot_id=None):
        self.pushed.append((int(frequency), channel, int(level)))

    def set_strength(self, channel, value, slot_id=None):
        self.strength_calls.append((channel, int(value)))
        async def _noop():
            pass
        return _noop()

    def set_wave(self, channel, name, slot_id=None):
        async def _noop():
            pass
        return _noop()

    def zap(self, channel, seconds=1.0, slot_id=None):
        async def _noop():
            pass
        return _noop()

    def fire_start(self, slot_id=None, channel=None):
        async def _noop():
            pass
        return _noop()

    def fire_stop(self, slot_id=None, channel=None):
        async def _noop():
            pass
        return _noop()

    def emergency_stop(self):
        async def _noop():
            pass
        return _noop()


def _bridge(config: dict | None = None) -> tuple[SoundBridge, FakeCommands]:
    """带哨兵输入流的桥接器（测试不触碰 sounddevice / 音频硬件）。"""
    commands = FakeCommands()
    bridge = SoundBridge(SoundConfig(config or {}), commands.get_state,
                         commands)
    bridge.log = lambda msg: None
    bridge._stream = object()                      # 哨兵：跳过真实开流
    bridge._opened_sig = bridge._stream_signature()
    bridge._samplerate = SR
    return bridge, commands


class AnalysisFunctionTests(unittest.TestCase):
    def test_rms_dbfs(self):
        self.assertEqual(rms_dbfs(np.zeros(100, dtype=np.float32)), -120.0)
        # 0.5 幅度正弦 RMS = 0.5/√2 ≈ -9.03 dBFS
        db = rms_dbfs(_sine(440, amp=0.5)[:, 0])
        self.assertAlmostEqual(db, -9.03, delta=0.1)

    def test_dbfs_to_level(self):
        self.assertEqual(dbfs_to_level(-60, -60, -10), 0.0)
        self.assertEqual(dbfs_to_level(-10, -60, -10), 100.0)
        self.assertEqual(dbfs_to_level(-35, -60, -10), 50.0)
        # 越界钳制
        self.assertEqual(dbfs_to_level(0, -60, -10), 100.0)
        self.assertEqual(dbfs_to_level(-120, -60, -10), 0.0)

    def test_dominant_frequency_hits_tone(self):
        window = _sine(440)[:, 0]
        freq = dominant_frequency(window, SR, 20.0, 2000.0)
        self.assertAlmostEqual(freq, 440.0, delta=3.0)
        window = _sine(880)[:, 1]
        freq = dominant_frequency(window, SR, 20.0, 2000.0)
        self.assertAlmostEqual(freq, 880.0, delta=3.0)

    def test_dominant_frequency_band_limited(self):
        # 带外能量（100Hz + 8kHz）不干扰带内主频 440
        t = np.arange(int(SR * 0.1)) / SR
        mix = (0.1 * np.sin(2 * math.pi * 100 * t)
               + 0.4 * np.sin(2 * math.pi * 440 * t)
               + 0.3 * np.sin(2 * math.pi * 8000 * t))
        freq = dominant_frequency(mix.astype(np.float32), SR, 200.0, 2000.0)
        self.assertAlmostEqual(freq, 440.0, delta=5.0)

    def test_dominant_frequency_silence_and_short(self):
        self.assertEqual(dominant_frequency(np.zeros(BLOCK, np.float32),
                                            SR, 20, 2000), 0.0)
        self.assertEqual(dominant_frequency(np.zeros(10, np.float32),
                                            SR, 20, 2000), 0.0)

    def test_hz_to_logical_log_scale(self):
        self.assertEqual(hz_to_logical(20, 20, 2000), 10)
        self.assertEqual(hz_to_logical(2000, 20, 2000), 1000)
        # 对数刻度中点：几何均值 200Hz → 10 + 990/2 ≈ 505
        self.assertEqual(hz_to_logical(200, 20, 2000), 505)
        # 越界钳制
        self.assertEqual(hz_to_logical(1, 20, 2000), 10)
        self.assertEqual(hz_to_logical(9000, 20, 2000), 1000)


class BridgeTickTests(unittest.IsolatedAsyncioTestCase):
    async def test_tick_feeds_four_variables_and_pushes_pulse(self):
        bridge, commands = _bridge()
        bridge._blocks.append(_sine(0, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()

        for side, freq in (("left", 440.0), ("right", 880.0)):
            self.assertAlmostEqual(bridge.engine.signals[f"{side}_frequency"],
                                   freq, delta=3.0)
            level = bridge.engine.signals[f"{side}_loudness"]
            self.assertGreater(level, 0.0)
            self.assertLessEqual(level, 100.0)
        # 两个通道各推一帧：频率对数映射 + 响度电平
        self.assertEqual(len(bridge.last_pushes), 2)
        by_ch = {ch: (f, lv) for f, ch, lv in bridge.last_pushes}
        self.assertEqual(by_ch["A"][0], hz_to_logical(440.0, 20.0, 2000.0))
        self.assertEqual(by_ch["B"][0], hz_to_logical(880.0, 20.0, 2000.0))
        self.assertGreater(by_ch["A"][1], 0)

    async def test_tick_without_selection_no_push(self):
        bridge, _commands = _bridge()
        _commands.selection = {"A": "__SILENT__", "B": PULSE_STREAM}
        bridge._blocks.append(_sine(440))
        bridge._tick()
        self.assertEqual([ch for _f, ch, _lv in bridge.last_pushes], ["B"])

    async def test_silence_pushes_zero_level_keeps_last_freq(self):
        bridge, _commands = _bridge()
        bridge._blocks.append(_sine(440))
        bridge._tick()
        last_logical = bridge.last_pushes[0][0]
        bridge._blocks.append(np.zeros((BLOCK, 2), dtype=np.float32))
        bridge._tick()
        self.assertEqual(len(bridge.last_pushes), 2)
        first, second = bridge.last_pushes
        self.assertEqual(second[2], 0)                       # 静音帧电平 0
        self.assertEqual(second[0], last_logical)            # 频率沿用上一拍
        self.assertEqual(first[2], 0)

    async def test_empty_window_still_pushes_silent_frame(self):
        bridge, _commands = _bridge()
        bridge._tick()
        self.assertEqual(len(bridge.last_pushes), 2)
        self.assertTrue(all(lv == 0 for _f, _c, lv in bridge.last_pushes))

    async def test_default_mappings_drive_strength(self):
        bridge, commands = _bridge()
        bridge._blocks.append(_sine(0, amp=0.5, per_channel={0: 440, 1: 880}))
        bridge._tick()
        # 默认行：{left_loudness} * 2 / {right_loudness} * 2 → 强度派发
        self.assertEqual(bridge.engine.mappings.get("in_strength_a"),
                         "{left_loudness} * 2")
        self.assertEqual(len(commands.strength_calls), 2)
        channels = {ch for ch, _v in commands.strength_calls}
        self.assertEqual(channels, {"A", "B"})
        for ch, value in commands.strength_calls:
            self.assertEqual(value, int(round(
                bridge.engine.signals[f"{'left' if ch == 'A' else 'right'}_loudness"]
                * 2)))

    async def test_configured_mappings_override_defaults(self):
        bridge, _ = _bridge({"mappings": [
            {"param": "in_strength_a", "expr": "{left_loudness}"},
        ]})
        self.assertEqual(bridge.engine.mappings.get("in_strength_a"),
                         "{left_loudness}")
        self.assertNotIn("in_strength_b", bridge.engine.mappings)

    async def test_reload_config_hot_swaps_mappings(self):
        bridge, _ = _bridge()
        self.assertEqual(bridge.engine.mappings.get("in_strength_a"),
                         "{left_loudness} * 2")
        bridge.config["mappings"] = [
            {"param": "in_strength_a", "expr": "{left_loudness} * 3"},
        ]
        await bridge.reload_config()
        self.assertEqual(bridge.engine.mappings.get("in_strength_a"),
                         "{left_loudness} * 3")

    async def test_swap_channels_routes_device_channels(self):
        bridge, _ = _bridge({"swap_channels": True})
        bridge._blocks.append(_sine(0, amp=0.5, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()
        # 默认左声道→A；交换后左声道数据推入 B 通道（变量名仍按音频声道）
        by_ch = {ch: f for f, ch, _lv in bridge.last_pushes}
        self.assertEqual(by_ch["A"], hz_to_logical(880.0, 20.0, 2000.0))
        self.assertEqual(by_ch["B"], hz_to_logical(440.0, 20.0, 2000.0))
        self.assertAlmostEqual(bridge.engine.signals["left_frequency"], 440.0,
                               delta=3.0)


class PluginContractTests(unittest.TestCase):
    def _plugin_source(self) -> str:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "modules", "sound_link", "plugin.py")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_meta_is_literal_and_complete(self):
        tree = ast.parse(self._plugin_source())
        meta = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "META"
                    for t in node.targets):
                meta = ast.literal_eval(node.value)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["id"], "sound_link")
        self.assertEqual(meta["settings_key"], "sound_link")
        self.assertEqual(meta["version"], "0.2.2")
        # 四个映射变量与 bridge PARAM_DEFS 一致
        self.assertEqual(set(meta["params"]), set(PARAM_DEFS))
        self.assertEqual(set(meta["params"]),
                         {"left_loudness", "right_loudness",
                          "left_frequency", "right_frequency"})
        # 配置声明：设备键 + 左右交换 + 输入映射表；纯输入模块无输出映射表
        cfg = meta["config"]
        self.assertIn("microphone", cfg)
        self.assertIn("speaker", cfg)
        self.assertIn("swap_channels", cfg)
        self.assertNotIn("device", cfg)          # 旧键已由两个设备键取代
        self.assertNotIn("outputs", cfg)         # 纯输入设计：无回传通道
        self.assertEqual(cfg["mappings"].get("rows"), "in")

    def test_config_spec_scans_devices_into_choices(self):
        from modules.sound_link import bridge as bridge_mod

        module = SoundLinkModule()
        spec = module.config_spec()
        # 未打补丁时依赖真实扫描：类型应为 choice 且首项为系统默认
        self.assertEqual(spec["microphone"]["type"], "choice")
        self.assertEqual(spec["speaker"]["type"], "choice")
        self.assertEqual(spec["microphone"]["choices"][0], "")
        self.assertEqual(spec["speaker"]["choices"][0], "")

        # 打补丁模拟扫描结果：choices = 系统默认 + 枚举设备名
        orig_mic, orig_spk = bridge_mod.list_microphones, bridge_mod.list_speakers
        try:
            bridge_mod.list_microphones = lambda: ["阵列麦克风 (AMD Audio Device)",
                                                   "麦克风 (PicoStreamingMicrophone)"]
            bridge_mod.list_speakers = lambda: ["扬声器 (Realtek(R) Audio)"]
            spec = module.config_spec()
            self.assertEqual(spec["microphone"]["choices"],
                             ["", "阵列麦克风 (AMD Audio Device)",
                              "麦克风 (PicoStreamingMicrophone)"])
            self.assertEqual(spec["speaker"]["choices"],
                             ["", "扬声器 (Realtek(R) Audio)"])
        finally:
            bridge_mod.list_microphones = orig_mic
            bridge_mod.list_speakers = orig_spk

    def test_bound_name_prefers_source_key_over_legacy(self):
        bridge, _ = _bridge({"microphone": "阵列麦克风", "device": "旧值"})
        self.assertEqual(bridge._bound_name(), "阵列麦克风")
        bridge2, _ = _bridge({"source": "loopback", "speaker": "", "device": "旧值"})
        self.assertEqual(bridge2._bound_name(), "旧值")

    def test_link_params_returns_four_variables(self):
        module = SoundLinkModule()
        params = module.link_params()
        self.assertEqual([name for name, _label in params],
                         ["left_loudness", "right_loudness",
                          "left_frequency", "right_frequency"])

    def test_default_mappings_target_core_inputs(self):
        from dglab.params import input_specs

        specs = input_specs()
        for row in DEFAULT_MAPPINGS:
            self.assertIn(row["param"], specs)
        self.assertEqual(DEFAULT_MAPPINGS[0]["expr"], "{left_loudness} * 2")
        self.assertEqual(DEFAULT_MAPPINGS[1]["expr"], "{right_loudness} * 2")

    def test_module_class_attributes(self):
        module = SoundLinkModule()
        self.assertEqual(module.id, "sound_link")
        self.assertEqual(module.settings_key, "sound_link")
        self.assertFalse(module.is_running())
        self.assertTrue(callable(module.config_spec))


if __name__ == "__main__":
    unittest.main()
