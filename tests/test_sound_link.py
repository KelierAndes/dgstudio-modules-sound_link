from __future__ import annotations

import ast
import math
import os
import unittest

import numpy as np

import _bootstrap  # noqa: F401  定位核心仓库并挂 sys.path

from dglab.state import EngineState, Slot

from modules.sound_link.bridge import (DEVICES, PARAM_DEFS, SoundBridge,
                                       SoundConfig, dbfs_to_level,
                                       dominant_frequency, hz_to_logical,
                                       migrate_settings, rms_dbfs)
from modules.sound_link.plugin import META, SoundLinkModule

SR = 48000.0
BLOCK = 4800

REMOVED_DEVICE_KEYS = ("mic_pulse", "loop_pulse", "mic_mappings", "loop_mappings")


def _sine(freq: float, seconds: float = 0.1, amp: float = 0.5,
          channels: int = 2, per_channel: dict[int, float] | None = None
          ) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    cols = []
    for ch in range(channels):
        f = (per_channel or {}).get(ch, freq)
        cols.append(amp * np.sin(2 * math.pi * f * t))
    return np.stack(cols, axis=1).astype(np.float32)


class ForbiddenCommands:
    """引擎命令层假件：任何设备直写调用都立即抛错，用来锁定「模块绝不驱动设备」。"""

    def __init__(self):
        self.state = EngineState(backend="ble")
        self.state.slots["s1"] = Slot(slot_id="s1", name="t", type="COYOTE_030")
        self.calls: list[str] = []

    def get_state(self):
        return self.state

    def _boom(self, name):
        self.calls.append(name)
        raise AssertionError(f"模块不应直写设备：{name}()")

    def set_strength(self, *a, **k):
        return self._boom("set_strength")

    def add_strength(self, *a, **k):
        return self._boom("add_strength")

    def reset_strength(self, *a, **k):
        return self._boom("reset_strength")

    def set_wave(self, *a, **k):
        return self._boom("set_wave")

    def push_pulse_stream(self, *a, **k):
        return self._boom("push_pulse_stream")

    def fire(self, *a, **k):
        return self._boom("fire")

    def fire_start(self, *a, **k):
        return self._boom("fire_start")

    def fire_stop(self, *a, **k):
        return self._boom("fire_stop")

    def zap(self, *a, **k):
        return self._boom("zap")

    def set_intensity_param(self, *a, **k):
        return self._boom("set_intensity_param")


def _bridge(config: dict | None = None, open_devices=("mic",)
            ) -> tuple[SoundBridge, ForbiddenCommands]:
    commands = ForbiddenCommands()
    bridge = SoundBridge(SoundConfig(config or {}), commands.get_state, commands)
    bridge.log = lambda msg: None
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

    def test_param_defs_cover_both_devices(self):
        self.assertEqual(len(PARAM_DEFS), 8)
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
        # 迁移不再新增推流 / 映射表配置项
        for key in REMOVED_DEVICE_KEYS:
            self.assertNotIn(key, settings)
        self.assertTrue(logs)

    def test_microphone_source_keeps_mic_only(self):
        settings = {"source": "microphone", "microphone": "阵列麦克风"}
        self.assertTrue(migrate_settings(settings, None))
        self.assertTrue(settings["mic_enabled"])
        self.assertFalse(settings["loop_enabled"])

    def test_removed_device_keys_are_dropped(self):
        logs: list[str] = []
        settings = {"mic_enabled": True, "mic_pulse": "ab",
                    "loop_pulse": "b", "mic_mappings": [{"param": "in_x"}],
                    "loop_mappings": []}
        self.assertTrue(migrate_settings(settings, logs.append))
        for key in REMOVED_DEVICE_KEYS:
            self.assertNotIn(key, settings)
        self.assertTrue(any("事件流" in m for m in logs))

    def test_source_and_removed_keys_migrate_together(self):
        settings = {"source": "loopback", "mic_pulse": "ab", "loop_mappings": [1]}
        self.assertTrue(migrate_settings(settings, None))
        self.assertFalse(settings["mic_enabled"])
        self.assertTrue(settings["loop_enabled"])
        self.assertNotIn("mic_pulse", settings)
        self.assertNotIn("loop_mappings", settings)

    def test_already_migrated_settings_untouched(self):
        settings = {"mic_enabled": False, "loop_enabled": True}
        self.assertFalse(migrate_settings(settings, None))
        self.assertEqual(settings, {"mic_enabled": False, "loop_enabled": True})


class BridgeSignalTests(unittest.TestCase):

    def test_tick_feeds_mic_variables(self):
        bridge, commands = _bridge()
        bridge.sources["mic"].blocks.append(
            _sine(0, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()
        signals = bridge.engine.signals
        for side, freq in (("left", 440.0), ("right", 880.0)):
            self.assertAlmostEqual(signals[f"mic_{side}_frequency"], freq,
                                   delta=3.0)
            level = signals[f"mic_{side}_loudness"]
            self.assertGreater(level, 0.0)
            self.assertLessEqual(level, 100.0)
        # 采集值只登记为变量；不触碰任何设备命令
        self.assertEqual(commands.calls, [])

    def test_disabled_device_reports_zero_and_stays_closed(self):
        bridge, _ = _bridge()
        bridge._tick()
        signals = bridge.engine.signals
        self.assertEqual(signals["loop_left_frequency"], 0.0)
        self.assertEqual(signals["loop_left_loudness"], 0.0)
        self.assertIsNone(bridge.sources["loop"].stream)

    def test_both_devices_publish_their_own_signals(self):
        bridge, commands = _bridge({"mic_enabled": True, "loop_enabled": True},
                                   open_devices=("mic", "loop"))
        bridge.sources["mic"].blocks.append(_sine(0, per_channel={0: 440.0}))
        bridge.sources["loop"].blocks.append(
            _sine(0, amp=0.4, per_channel={1: 880.0}))
        bridge._tick()
        signals = bridge.engine.signals
        self.assertAlmostEqual(signals["mic_left_frequency"], 440.0, delta=3.0)
        self.assertAlmostEqual(signals["loop_right_frequency"], 880.0, delta=3.0)
        self.assertAlmostEqual(signals["loop_left_loudness"], 0.0)
        self.assertEqual(commands.calls, [])

    def test_silence_zeroes_signals(self):
        bridge, commands = _bridge()
        bridge.sources["mic"].blocks.append(_sine(0, per_channel={0: 440.0}))
        bridge._tick()
        self.assertGreater(bridge.engine.signals["mic_left_loudness"], 0.0)
        bridge.sources["mic"].blocks.append(np.zeros((BLOCK, 2), np.float32))
        bridge._tick()
        signals = bridge.engine.signals
        self.assertEqual(signals["mic_left_loudness"], 0.0)
        self.assertEqual(commands.calls, [])

    def test_swap_channels_swaps_audio_columns(self):
        bridge, _ = _bridge({"swap_channels": True})
        bridge.sources["mic"].blocks.append(
            _sine(0, amp=0.5, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()
        signals = bridge.engine.signals
        self.assertAlmostEqual(signals["mic_left_frequency"], 880.0, delta=3.0)
        self.assertAlmostEqual(signals["mic_right_frequency"], 440.0, delta=3.0)

    def test_merged_signals_and_last_values_match(self):
        bridge, _ = _bridge({"loop_enabled": True}, open_devices=("mic", "loop"))
        bridge.sources["loop"].blocks.append(_sine(0, per_channel={0: 300.0}))
        bridge._tick()
        self.assertIn("mic_left_loudness", bridge.engine.signals)
        self.assertIn("loop_left_frequency", bridge.engine.signals)
        self.assertEqual(sorted(bridge.last_values),
                         sorted(bridge.engine.signals))

    def test_engine_exposes_signals_errors_and_pump(self):
        bridge, _ = _bridge()
        engine = bridge.engine
        self.assertIsInstance(engine.signals, dict)
        self.assertIsInstance(engine.errors, dict)
        self.assertIsNone(engine.pump())

    def test_source_keeps_plain_signals_dict(self):
        bridge, _ = _bridge()
        bridge._tick()
        source = bridge.sources["mic"]
        self.assertIsInstance(source.signals, dict)
        self.assertNotIn("engine", vars(source))


class NoDeviceCommandPathTests(unittest.TestCase):

    def test_bridge_has_no_device_write_paths(self):
        bridge, _ = _bridge()
        for attr in ("dispatchers", "_push_pulse", "_pulse_owner", "_DeviceApi",
                     "_dispatch", "_device_vars", "apply_config", "_api"):
            self.assertFalse(hasattr(bridge, attr), f"残留设备写路径: {attr}")

    def test_tick_and_start_never_call_device_commands(self):
        bridge, commands = _bridge(open_devices=("mic",))
        bridge.sources["mic"].blocks.append(_sine(0, per_channel={0: 440.0, 1: 880.0}))
        bridge._tick()
        self.assertEqual(commands.calls, [])

    def test_bridge_module_has_no_device_command_names(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "modules", "sound_link", "bridge.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        for forbidden in ("set_strength", "push_pulse_stream", "build_dispatchers",
                          "core_inputs", "MappingEngine", "set_mappings",
                          "fire_start", "zap"):
            self.assertNotIn(forbidden, src, f"bridge.py 仍引用设备写符号: {forbidden}")


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
        self.assertEqual(meta["version"], "0.5.1")
        cfg = meta["config"]
        self.assertEqual(set(cfg), {"mic_enabled", "microphone", "loop_enabled",
                                    "speaker", "swap_channels", "gain", "min_db",
                                    "max_db", "smooth", "min_hz", "max_hz"})
        for key in REMOVED_DEVICE_KEYS:
            self.assertNotIn(key, cfg)
        self.assertNotIn("source", cfg)
        self.assertFalse(cfg["loop_enabled"]["default"])
        self.assertTrue(cfg["mic_enabled"]["default"])

    def test_meta_description_is_pure_input(self):
        self.assertIn("只读", META["description"])
        self.assertIn("事件流", META["description"])

    def test_params_follow_param_defs(self):
        self.assertEqual(set(META["params"]), set(PARAM_DEFS))

    def test_link_params_are_readonly_float_dicts(self):
        module = SoundLinkModule()
        params = module.link_params()
        self.assertEqual([p["name"] for p in params], list(PARAM_DEFS))
        self.assertTrue(all(p["dir"] == "in" for p in params))
        self.assertTrue(all(p["type"] == "Float" for p in params))

    def test_temp_specs_list_eight_readable_vars(self):
        module = SoundLinkModule()
        specs = {spec["key"]: spec for spec in module.temp_specs()}
        self.assertEqual(set(specs), set(PARAM_DEFS))
        self.assertTrue(all(spec["dir"] == "in" for spec in specs.values()))
        self.assertTrue(all(spec["type"] == "Float"
                            for spec in specs.values()))

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

    def test_on_load_migrates_drops_removed_keys_and_events(self):
        module = SoundLinkModule()
        logs: list[str] = []

        class FakeSettings(dict):
            saved = 0

            def save(self):
                FakeSettings.saved += 1

        class FakeCtx:
            engine = None
            events = None

            def log(self, msg):
                logs.append(msg)

        ctx = FakeCtx()
        ctx.settings = FakeSettings(source="loopback", mic_pulse="ab",
                                    loop_mappings=[{"param": "in_x"}],
                                    events=[{"name": "旧事件卡"}])
        module.on_load(ctx)
        self.assertNotIn("source", ctx.settings)
        self.assertTrue(ctx.settings["loop_enabled"])
        self.assertNotIn("mic_pulse", ctx.settings)
        self.assertNotIn("loop_mappings", ctx.settings)
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
