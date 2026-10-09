META = {
    "id": "sound_link",
    "name": "音频联动",
    "version": "0.4.0",
    "description": "纯输入联动：麦克风与系统声音（WASAPI 回环）双路同时监听，"
                   "每路各输出左右响度 / 左右频率八个映射变量、各维护一张映射表；"
                   "频率值直接推入核心「外部脉冲流」参数（0.1s 一拍），输出频率跟随声音音高。",
    "settings_key": "sound_link",
    "default_enabled": False,
    "params": {},
    "config": {
        "mic_enabled": {
            "label": "监听麦克风", "type": "bool", "default": True,
            "group": "audio",
            "desc": "开启后采集麦克风输入（与系统声音可同时开启）",
        },
        "microphone": {
            "label": "麦克风设备", "type": "str", "default": "",
            "group": "audio",
            "desc": "设备名由模块自动扫描枚举（模块页为下拉选择框）；"
                    "留空用系统默认录音设备",
        },
        "mic_pulse": {
            "label": "麦克风 → 脉冲流", "type": "choice",
            "choices": ["off", "ab", "a", "b"],
            "labels": ["不推流", "左→A · 右→B", "仅左→A", "仅右→B"],
            "default": "ab", "group": "audio",
            "desc": "把麦克风左右声道的频率值按对数刻度推入核心「外部脉冲流」；"
                    "静音时推 0（停止脉冲）",
        },
        "mic_mappings": {
            "label": "麦克风映射表", "type": "list", "default": [],
            "group": "map", "rows": "in",
            "desc": "行 {param: 核心输入参数, expr: 表达式}，变量用 "
                    "{mic_left_loudness} {mic_left_frequency} 等本路采集值，"
                    "结果取整钳制后派发设备动作",
        },
        "loop_enabled": {
            "label": "监听系统声音", "type": "bool", "default": False,
            "group": "audio",
            "desc": "Windows WASAPI 回环采集正在播放的声音；与麦克风互不影响，"
                    "两路各自一张映射表（改动后重新开关模块生效）",
        },
        "speaker": {
            "label": "播放设备（回环）", "type": "str", "default": "",
            "group": "audio",
            "desc": "回环采集的播放设备（自动枚举）；留空用系统默认播放设备",
        },
        "loop_pulse": {
            "label": "系统声音 → 脉冲流", "type": "choice",
            "choices": ["off", "ab", "a", "b"],
            "labels": ["不推流", "左→A · 右→B", "仅左→A", "仅右→B"],
            "default": "off", "group": "audio",
            "desc": "两路同时推同一通道会互相抢驱动，通常只开一路推流",
        },
        "loop_mappings": {
            "label": "系统声音映射表", "type": "list", "default": [],
            "group": "map", "rows": "in",
            "desc": "变量用 {loop_left_loudness} {loop_left_frequency} 等本路采集值",
        },
        "swap_channels": {
            "label": "左右声道交换", "type": "bool", "default": False,
            "group": "audio",
            "desc": "默认采集声道0(物理左)→左变量、声道1(物理右)→右变量；"
                    "开启后交换（两路采集同时生效，现场左右接反时对调）",
        },
        "gain": {
            "label": "响度增益", "type": "float", "default": 1.0,
            "min": 0.1, "max": 20.0, "step": 0.1, "group": "audio",
            "desc": "dB 映射前乘系数，声音偏小可调大",
        },
        "min_db": {
            "label": "响度 0% (dBFS)", "type": "float", "default": -60.0,
            "min": -90.0, "max": -1.0, "step": 1.0, "group": "audio",
            "desc": "低于该 RMS 电平响度记 0",
        },
        "max_db": {
            "label": "响度 100% (dBFS)", "type": "float", "default": -10.0,
            "min": -60.0, "max": 0.0, "step": 1.0, "group": "audio",
            "desc": "高于该 RMS 电平响度记 100",
        },
        "smooth": {
            "label": "响度平滑", "type": "float", "default": 0.5,
            "min": 0.0, "max": 0.95, "step": 0.05, "group": "audio",
            "desc": "指数平滑系数，越大越稳（0 关闭平滑）",
        },
        "min_hz": {
            "label": "音高下限 (Hz)", "type": "int", "default": 20,
            "min": 10, "max": 500, "step": 5, "group": "audio",
            "desc": "检测下限，映射到设备频率 10（对数刻度）",
        },
        "max_hz": {
            "label": "音高上限 (Hz)", "type": "int", "default": 1000,
            "min": 100, "max": 1000, "step": 10, "group": "audio",
            "desc": "检测上限，映射到设备频率 1000（对数刻度）",
        },
    },
}

from plugins import ModuleBase, spec_defaults

from modules.sound_link import bridge as _bridge_mod
from modules.sound_link.bridge import (PARAM_DEFS, SoundBridge,
                                       SoundConfig, migrate_settings)

META["params"] = {name: dict(item)
                 for name, item in PARAM_DEFS.items()}
SOUND_CONFIG_DEFAULTS = spec_defaults(META["config"])

_DEVICE_ENUMS = {
    "microphone": "list_microphones",
    "speaker": "list_speakers",
}


class SoundLinkModule(ModuleBase):
    id = META["id"]
    name = META["name"]
    version = META["version"]
    description = META["description"]
    settings_key = META["settings_key"]

    def __init__(self):
        self.bridge: SoundBridge | None = None
        self.ctx = None

    def config_spec(self) -> dict:
        spec = dict(META["config"])
        for key, enum_name in _DEVICE_ENUMS.items():
            item = dict(spec.get(key) or {})
            names = []
            try:
                names = list(getattr(_bridge_mod, enum_name)())
            except Exception:
                names = []
            item["type"] = "choice"
            item["choices"] = [""] + list(names)
            spec[key] = item
        return spec

    def link_params(self) -> list[tuple[str, str]]:
        return [(name, str(item.get("label") or ""))
                for name, item in PARAM_DEFS.items()]

    def temp_specs(self) -> list[dict]:
        """采集值由本模块每拍维护：只读，回传方向没有意义。"""
        return [{"key": name, "label": str(item.get("label") or ""),
                 "dir": "in", "desc": str(item.get("desc") or "")}
                for name, item in PARAM_DEFS.items()]

    def on_load(self, ctx) -> None:
        self.ctx = ctx
        if migrate_settings(ctx.settings, ctx.log) and hasattr(ctx.settings, "save"):
            ctx.settings.save()
        ctx.settings.pop("events", None)   # 推流改由模块直接驱动，不再播种事件卡

    def on_unload(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
        self.bridge = None
        self.ctx = None

    async def start(self) -> None:
        if self.bridge is not None and self.bridge._running:
            return
        if self.bridge is not None:
            try:
                await self.bridge.stop()
            except Exception:
                pass
        self.bridge = SoundBridge(
            SoundConfig(self.ctx.settings, defaults=SOUND_CONFIG_DEFAULTS),
            self.ctx.engine.get_state,
            self.ctx.engine,
            events=self.ctx.events,
        )
        self.bridge.log = self.ctx.log
        await self.bridge.start()

    async def reload_config(self) -> None:
        if self.bridge is None:
            return
        for key in SOUND_CONFIG_DEFAULTS:
            if key in self.ctx.settings:
                self.bridge.config[key] = self.ctx.settings[key]
        self.bridge.apply_config()

    async def stop(self) -> None:
        if self.bridge is not None:
            await self.bridge.stop()

    def is_running(self) -> bool:
        return self.bridge is not None and bool(getattr(self.bridge,
                                                        "_running", False))
