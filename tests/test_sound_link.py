"""音频联动模块单测。

覆盖：分析纯函数（响度 dB 映射 / FFT 主频 / 频率→设备逻辑频率对数映射）、
桥接器一拍分析喂六个信号（响度/频率/推流值 × 左右）、事件流周期卡推入
核心 in_pulse_* 参数（经核心派发器，0=静音帧、0.1s 节流）、默认事件卡
播种、插件 META / link_params 契约。不依赖音频硬件（sounddevice /
pyaudiowpatch 仅在真实开流时导入，测试用哨兵流对象绕过）。

运行（模块仓库根目录）::

    python -m unittest discover -s tests
"""
from __future__ import annotations

import asyncio
import ast
import math
import os
import unittest
import unittest.mock

import numpy as np

import _bootstrap  # noqa: F401  定位核心仓库并挂 sys.path

import dglab.params as params_mod
from dglab.params import input_specs
from dglab.state import EngineState, Slot

from modules.sound_link.bridge import (DEFAULT_EVENT_CARDS, PARAM_DEFS,
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
    """引擎命令层假件：记录事件流派发链路上的脉冲推入（不触碰真实设备）。"""

    def __init__(self):
        self.state = EngineState(backend="ble")
        self.state.slots["s1"] = Slot(slot_id="s1", name="t", type="COYOTE_030")
        self.pushed: list[tuple[int, str, int]] = []

    def get_state(self):
        return self.state

    async def push_pulse_stream(self, frequency, channel="A", level=100,
                                slot_id=None):
        self.pushed.append((int(frequency), channel, int(level)))

    def set_strength(self, channel, value, slot_id=None):
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
    """一拍分析：喂六个信号；脉冲推入全部经事件流派发链路。"""

    def setUp(self):
        # 派发器 0.1s 节流旁路：测试快速连拍不丢帧
        patcher = unittest.mock.patch.object(params_mod,
                                             "PULSE_PUSH_MIN_INTERVAL_S", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_tick_feeds_six_variables(self):
        bridge, commands = _bridge()
        bridge._blocks.append(_sine(0, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()

        for side, freq in (("left", 440.0), ("right", 880.0)):
            self.assertAlmostEqual(bridge.engine.signals[f"{side}_frequency"],
                                   freq, delta=3.0)
            level = bridge.engine.signals[f"{side}_loudness"]
            self.assertGreater(level, 0.0)
            self.assertLessEqual(level, 100.0)
            # 推流值 = 主频对数映射到设备逻辑频率
            self.assertEqual(bridge.engine.signals[f"{side}_pulse"],
                             hz_to_logical(freq, 20.0, 2000.0))
        # 未配置事件流（宿主未装载卡片）时模块自身不推帧
        self.assertEqual(commands.pushed, [])

    async def test_event_card_pushes_pulse_frames(self):
        """默认事件卡（周期 100ms）把推流值推入 in_pulse_a/b。"""
        bridge, commands = _bridge()
        bridge._loop = asyncio.get_running_loop()   # _spawn 调度推流协程用
        bridge.engine.set_event_cards(DEFAULT_EVENT_CARDS)
        bridge._blocks.append(_sine(0, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()
        bridge.engine.tick_event_cards(0.0)       # 首拍立即到期
        await asyncio.sleep(0)                    # 让出节拍：推流协程执行
        self.assertEqual(len(commands.pushed), 2)
        by_ch = {ch: (f, lv) for f, ch, lv in commands.pushed}
        self.assertEqual(by_ch["A"][0], hz_to_logical(440.0, 20.0, 2000.0))
        self.assertEqual(by_ch["B"][0], hz_to_logical(880.0, 20.0, 2000.0))
        self.assertEqual(by_ch["A"][1], 100)      # 非静音帧电平 100

        # 恒定音：周期再次到期时同值也继续推帧（流语义，非边沿动作）
        bridge._tick()
        bridge.engine.tick_event_cards(0.1)
        await asyncio.sleep(0)
        self.assertEqual(len(commands.pushed), 4)

    async def test_silence_pushes_zero_pulse(self):
        """静音：推流值归 0 → 派发链路生成电平 0 的静音帧，频率沿用上一拍。"""
        bridge, commands = _bridge()
        bridge._loop = asyncio.get_running_loop()
        bridge.engine.set_event_cards(DEFAULT_EVENT_CARDS)
        bridge._blocks.append(_sine(440))
        bridge._tick()
        bridge.engine.tick_event_cards(0.0)
        await asyncio.sleep(0)
        audible_logical = commands.pushed[0][0]
        bridge._blocks.append(np.zeros((BLOCK, 2), dtype=np.float32))
        bridge._tick()
        bridge.engine.tick_event_cards(0.1)
        await asyncio.sleep(0)
        self.assertEqual(bridge.engine.signals["left_pulse"], 0)
        silent = [p for p in commands.pushed if p[1] == "A"][-1]
        self.assertEqual(silent[2], 0)                    # 静音帧电平 0
        self.assertGreater(audible_logical, 0)

    async def test_empty_window_feeds_zero_signals(self):
        bridge, _ = _bridge()
        bridge._tick()
        self.assertEqual(bridge.engine.signals["left_pulse"], 0)
        self.assertEqual(bridge.engine.signals["right_loudness"], 0.0)

    async def test_swap_channels_swaps_audio_columns(self):
        bridge, _ = _bridge({"swap_channels": True})
        bridge._blocks.append(_sine(0, amp=0.5, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()
        # 交换后：采集声道1(880) → 左变量、声道0(440) → 右变量
        self.assertAlmostEqual(bridge.engine.signals["left_frequency"], 880.0,
                               delta=3.0)
        self.assertAlmostEqual(bridge.engine.signals["right_frequency"], 440.0,
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
        self.assertEqual(meta["version"], "0.3.1")
        # 六个映射变量与 bridge PARAM_DEFS 一致
        self.assertEqual(set(meta["params"]), set(PARAM_DEFS))
        self.assertEqual(set(meta["params"]),
                         {"left_loudness", "right_loudness",
                          "left_frequency", "right_frequency",
                          "left_pulse", "right_pulse"})
        # 配置声明：设备键 + 左右交换；无映射表/输出表（事件流是唯一数据面）
        cfg = meta["config"]
        self.assertIn("microphone", cfg)
        self.assertIn("speaker", cfg)
        self.assertIn("swap_channels", cfg)
        self.assertNotIn("device", cfg)          # 旧键已由两个设备键取代
        self.assertNotIn("mappings", cfg)
        self.assertNotIn("outputs", cfg)

    def test_config_spec_scans_devices_into_choices(self):
        from modules.sound_link import bridge as bridge_mod

        module = SoundLinkModule()
        spec = module.config_spec()
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

    def test_link_params_returns_six_variables(self):
        module = SoundLinkModule()
        params = module.link_params()
        self.assertEqual([name for name, _label in params],
                         ["left_loudness", "right_loudness",
                          "left_frequency", "right_frequency",
                          "left_pulse", "right_pulse"])

    def test_default_event_cards_target_core_inputs(self):
        """默认事件卡动作引用的核心输入参数与变量全部有效。"""
        specs = input_specs()
        for card in DEFAULT_EVENT_CARDS:
            self.assertEqual(card["trigger"], "period")
            self.assertEqual(card["arg"], 100)
            for action in card["actions"]:
                self.assertEqual(action["dir"], "in")
                self.assertIn(action["param"], specs)
                self.assertEqual(specs[action["param"]]["action"], "pulse")
                self.assertIn(action["var"], PARAM_DEFS)

    def test_on_load_seeds_default_events_once(self):
        module = SoundLinkModule()
        logs: list[str] = []

        class FakeCtx:
            engine = None

            def log(self, msg):
                logs.append(msg)

        ctx = FakeCtx()
        ctx.settings = {}
        module.on_load(ctx)
        self.assertIn("events", ctx.settings)
        self.assertEqual(len(ctx.settings["events"]), 1)
        self.assertEqual(ctx.settings["events"][0]["name"], "音频脉冲流推入")
        self.assertTrue(logs)
        # 再次装载不覆盖用户编辑
        ctx.settings["events"] = [{"name": "自定义"}]
        module.on_load(ctx)
        self.assertEqual(ctx.settings["events"], [{"name": "自定义"}])

    def test_module_class_attributes(self):
        module = SoundLinkModule()
        self.assertEqual(module.id, "sound_link")
        self.assertEqual(module.settings_key, "sound_link")
        self.assertFalse(module.is_running())
        self.assertTrue(callable(module.config_spec))


if __name__ == "__main__":
    unittest.main()
