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
from dglab.state import EngineState, Slot

from modules.sound_link.bridge import (DEVICES, PARAM_DEFS, SoundBridge,
                                       SoundConfig, dbfs_to_level,
                                       dominant_frequency, hz_to_logical,
                                       migrate_settings, rms_dbfs)
from modules.sound_link.plugin import META, SoundLinkModule

SR = 48000.0
BLOCK = 4800


def _sine(freq: float, seconds: float = 0.1, amp: float = 0.5,
          channels: int = 2, per_channel: dict[int, float] | None = None
          ) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    cols = []
    for ch in range(channels):
        f = (per_channel or {}).get(ch, freq)
        cols.append(amp * np.sin(2 * math.pi * f * t))
    return np.stack(cols, axis=1).astype(np.float32)


class FakeCommands:

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


def _bridge(config: dict | None = None, open_devices=("mic",)
            ) -> tuple[SoundBridge, FakeCommands]:
    commands = FakeCommands()
    bridge = SoundBridge(SoundConfig(config or {}), commands.get_state,
                         commands)
    bridge.log = lambda msg: None
    try:
        bridge._loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    for key in open_devices:
        source = bridge.sources[key]
        if key == "loop":
            source.pa_stream = object()
        else:
            source.stream = object()
        source.opened_sig = source.signature()
        source.samplerate = SR
    return bridge, commands


class AnalysisFunctionTests(unittest.TestCase):
    def test_rms_dbfs(self):
        self.assertEqual(rms_dbfs(np.zeros(100, dtype=np.float32)), -120.0)
        db = rms_dbfs(_sine(440, amp=0.5)[:, 0])
        self.assertAlmostEqual(db, -9.03, delta=0.1)

    def test_dbfs_to_level(self):
        self.assertEqual(dbfs_to_level(-60, -60, -10), 0.0)
        self.assertEqual(dbfs_to_level(-10, -60, -10), 100.0)
        self.assertEqual(dbfs_to_level(-35, -60, -10), 50.0)
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
        self.assertEqual(hz_to_logical(200, 20, 2000), 505)
        self.assertEqual(hz_to_logical(1, 20, 2000), 10)
        self.assertEqual(hz_to_logical(9000, 20, 2000), 1000)

    def test_param_defs_cover_both_devices_without_pulse_vars(self):
        self.assertEqual(len(PARAM_DEFS), 8)
        self.assertNotIn("left_pulse", PARAM_DEFS)
        for key, _label in DEVICES:
            for name in (f"{key}_left_loudness", f"{key}_left_frequency",
                         f"{key}_right_loudness", f"{key}_right_frequency"):
                self.assertIn(name, PARAM_DEFS)


class MigrateSettingsTests(unittest.TestCase):

    def test_legacy_single_source_becomes_two_switches(self):
        logs: list[str] = []
        settings = {"source": "loopback", "speaker": "扬声器"}
        self.assertTrue(migrate_settings(settings, logs.append))
        self.assertNotIn("source", settings)
        self.assertFalse(settings["mic_enabled"])
        self.assertTrue(settings["loop_enabled"])
        self.assertEqual(settings["mic_pulse"], "off")
        self.assertEqual(settings["loop_pulse"], "ab")
        self.assertTrue(logs)

    def test_microphone_source_keeps_mic_only(self):
        settings = {"source": "microphone", "microphone": "阵列麦克风"}
        self.assertTrue(migrate_settings(settings, None))
        self.assertTrue(settings["mic_enabled"])
        self.assertFalse(settings["loop_enabled"])
        self.assertEqual(settings["mic_pulse"], "ab")

    def test_already_migrated_settings_untouched(self):
        settings = {"mic_enabled": False, "loop_enabled": True}
        self.assertFalse(migrate_settings(settings, None))
        self.assertEqual(settings, {"mic_enabled": False, "loop_enabled": True})


class BridgeTickTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        patcher = unittest.mock.patch.object(params_mod,
                                             "PULSE_PUSH_MIN_INTERVAL_S", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_tick_feeds_mic_variables(self):
        bridge, commands = _bridge()
        bridge.sources["mic"].blocks.append(
            _sine(0, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()
        await asyncio.sleep(0)
        signals = bridge.engine.signals
        for side, freq in (("left", 440.0), ("right", 880.0)):
            self.assertAlmostEqual(signals[f"mic_{side}_frequency"], freq,
                                   delta=3.0)
            level = signals[f"mic_{side}_loudness"]
            self.assertGreater(level, 0.0)
            self.assertLessEqual(level, 100.0)
            self.assertNotIn(f"mic_{side}_pulse", signals)
        # 频率直接驱动推流：一拍两通道
        by_ch = {ch: (f, lv) for f, ch, lv in commands.pushed}
        self.assertEqual(by_ch["A"][0], hz_to_logical(440.0, 20.0, 1000.0))
        self.assertEqual(by_ch["B"][0], hz_to_logical(880.0, 20.0, 1000.0))
        self.assertGreater(by_ch["A"][1], 0)

    async def test_disabled_device_reports_zero_and_stays_closed(self):
        bridge, _ = _bridge()
        bridge._tick()
        signals = bridge.engine.signals
        self.assertEqual(signals["loop_left_frequency"], 0.0)
        self.assertEqual(signals["loop_left_loudness"], 0.0)
        self.assertIsNone(bridge.sources["loop"].stream)

    async def test_both_devices_run_with_own_signal_sets(self):
        bridge, commands = _bridge({"mic_enabled": True, "loop_enabled": True,
                                    "mic_pulse": "a", "loop_pulse": "b"},
                                   open_devices=("mic", "loop"))
        bridge.sources["mic"].blocks.append(_sine(0, per_channel={0: 440.0}))
        bridge.sources["loop"].blocks.append(
            _sine(0, amp=0.4, per_channel={1: 880.0}))
        bridge._tick()
        await asyncio.sleep(0)
        signals = bridge.engine.signals
        self.assertAlmostEqual(signals["mic_left_frequency"], 440.0, delta=3.0)
        self.assertAlmostEqual(signals["loop_right_frequency"], 880.0, delta=3.0)
        self.assertAlmostEqual(signals["loop_left_loudness"], 0.0)
        channels = {ch for _f, ch, _lv in commands.pushed}
        self.assertEqual(channels, {"A", "B"})

    async def test_two_mapping_tables_are_independent(self):
        pushed: list[tuple[str, int]] = []
        bridge, _ = _bridge({"mic_mappings": [
            {"param": "in_strength_a", "expr": "{mic_left_loudness} * 2"}],
            "loop_mappings": [
            {"param": "in_strength_b", "expr": "{loop_right_loudness} * 3"}]},
            open_devices=("mic", "loop"))
        bridge.dispatchers = {
            "in_strength_a": lambda value: pushed.append(("A", value)),
            "in_strength_b": lambda value: pushed.append(("B", value)),
            "in_pulse_a": lambda value: None,
            "in_pulse_b": lambda value: None,
        }
        bridge.apply_config()
        self.assertEqual(bridge.sources["mic"].engine.mappings,
                         {"in_strength_a": "{mic_left_loudness} * 2"})
        self.assertEqual(bridge.sources["loop"].engine.mappings,
                         {"in_strength_b": "{loop_right_loudness} * 3"})
        bridge.sources["mic"].blocks.append(_sine(0, per_channel={0: 440.0}))
        bridge.sources["loop"].blocks.append(_sine(0, per_channel={1: 440.0}))
        bridge._tick()
        self.assertTrue(any(name == "A" and value > 0 for name, value in pushed))
        self.assertTrue(any(name == "B" and value > 0 for name, value in pushed))
        # 跨表引用不存在的变量时不派发（两路变量互不可见）
        bridge.sources["mic"].engine.set_mappings(
            [{"param": "in_strength_a", "expr": "{loop_left_loudness}"}])
        pushed.clear()
        bridge.sources["mic"].engine.signal("mic_left_loudness", 50.0)
        self.assertEqual(pushed, [])

    async def test_silence_pushes_zero_pulse(self):
        bridge, commands = _bridge()
        bridge.sources["mic"].blocks.append(_sine(440))
        bridge._tick()
        await asyncio.sleep(0)
        audible = [p for p in commands.pushed if p[1] == "A"][-1]
        bridge.sources["mic"].blocks.append(np.zeros((BLOCK, 2), np.float32))
        bridge._tick()
        await asyncio.sleep(0)
        silent = [p for p in commands.pushed if p[1] == "A"][-1]
        self.assertEqual(silent[2], 0)
        self.assertGreater(audible[0], 0)

    async def test_pulse_off_mode_pushes_nothing(self):
        bridge, commands = _bridge({"mic_pulse": "off"})
        bridge.sources["mic"].blocks.append(_sine(440))
        bridge._tick()
        self.assertEqual(commands.pushed, [])

    async def test_swap_channels_swaps_audio_columns(self):
        bridge, _ = _bridge({"swap_channels": True})
        bridge.sources["mic"].blocks.append(
            _sine(0, amp=0.5, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()
        signals = bridge.engine.signals
        self.assertAlmostEqual(signals["mic_left_frequency"], 880.0, delta=3.0)
        self.assertAlmostEqual(signals["mic_right_frequency"], 440.0, delta=3.0)

    async def test_merged_signals_reach_core_view(self):
        bridge, _ = _bridge({"loop_enabled": True}, open_devices=("mic", "loop"))
        bridge.sources["loop"].blocks.append(_sine(0, per_channel={0: 300.0}))
        bridge._tick()
        self.assertIn("mic_left_loudness", bridge.engine.signals)
        self.assertIn("loop_left_frequency", bridge.engine.signals)
        self.assertEqual(sorted(bridge.last_values),
                         sorted(bridge.engine.signals))


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
        self.assertEqual(meta["version"], "0.4.0")
        cfg = meta["config"]
        self.assertEqual(set(cfg), {"mic_enabled", "microphone", "mic_pulse",
                                    "mic_mappings", "loop_enabled", "speaker",
                                    "loop_pulse", "loop_mappings",
                                    "swap_channels", "gain", "min_db",
                                    "max_db", "smooth", "min_hz", "max_hz"})
        self.assertNotIn("source", cfg)
        self.assertFalse(cfg["loop_enabled"]["default"])
        self.assertTrue(cfg["mic_enabled"]["default"])
        self.assertEqual(cfg["mic_pulse"]["default"], "ab")
        self.assertEqual(cfg["loop_pulse"]["default"], "off")

    def test_params_follow_param_defs(self):
        self.assertEqual(set(META["params"]), set(PARAM_DEFS))

    def test_link_and_temp_specs_list_eight_readable_vars(self):
        module = SoundLinkModule()
        names = [name for name, _label in module.link_params()]
        self.assertEqual(names, list(PARAM_DEFS))
        specs = {spec["key"]: spec for spec in module.temp_specs()}
        self.assertEqual(set(specs), set(PARAM_DEFS))
        self.assertTrue(all(spec["dir"] == "in" for spec in specs.values()))

    def test_config_spec_scans_devices_into_choices(self):
        from modules.sound_link import bridge as bridge_mod

        module = SoundLinkModule()
        spec = module.config_spec()
        self.assertEqual(spec["microphone"]["type"], "choice")
        self.assertEqual(spec["speaker"]["type"], "choice")
        self.assertEqual(spec["microphone"]["choices"][0], "")
        self.assertEqual(spec["speaker"]["choices"][0], "")

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

    def test_on_load_migrates_and_drops_legacy_events(self):
        module = SoundLinkModule()
        logs: list[str] = []

        class FakeSettings(dict):
            saved = 0

            def save(self):
                FakeSettings.saved += 1

        class FakeCtx:
            engine = None

            def log(self, msg):
                logs.append(msg)

        ctx = FakeCtx()
        ctx.settings = FakeSettings(source="loopback",
                                    events=[{"name": "旧事件卡"}])
        module.on_load(ctx)
        self.assertNotIn("source", ctx.settings)
        self.assertTrue(ctx.settings["loop_enabled"])
        self.assertNotIn("events", ctx.settings)
        self.assertGreaterEqual(FakeSettings.saved, 1)

    def test_module_class_attributes(self):
        module = SoundLinkModule()
        self.assertEqual(module.id, "sound_link")
        self.assertEqual(module.settings_key, "sound_link")
        self.assertFalse(module.is_running())
        self.assertTrue(callable(module.config_spec))

    def test_device_names_are_per_source(self):
        bridge, _ = _bridge({"microphone": "阵列麦克风", "speaker": "扬声器"})
        self.assertEqual(bridge.sources["mic"].device_name(), "阵列麦克风")
        self.assertEqual(bridge.sources["loop"].device_name(), "扬声器")


if __name__ == "__main__":
    unittest.main()
