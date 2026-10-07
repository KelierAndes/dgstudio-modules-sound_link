
META = {
    "id": "sound_link",
    "name": "音频联动",
    "version": "0.3.2",
    "description": "纯输入联动：采集麦克风/系统声音，实时输出左/右响度、"
                   "左/右频率与左右推流值六个映射变量；事件流周期把推流值"
                   "推入核心「外部脉冲流」参数，输出频率跟随声音音高。",
    "settings_key": "sound_link",
    "default_enabled": False,
    "params": {
        "left_loudness": {"label": "左响度", "desc": "左声道响度 0-100"},
        "right_loudness": {"label": "右响度", "desc": "右声道响度 0-100"},
        "left_frequency": {"label": "左频率", "desc": "左声道主频率 (Hz)"},
        "right_frequency": {"label": "右频率", "desc": "右声道主频率 (Hz)"},
        "left_pulse": {"label": "左推流",
                       "desc": "左声道推流值（静音=0，否则=设备逻辑频率）"},
        "right_pulse": {"label": "右推流",
                        "desc": "右声道推流值（静音=0，否则=设备逻辑频率）"},
    },
    "config": {
        "source": {
            "label": "声音来源", "type": "choice",
            "choices": ["microphone", "loopback"], "default": "microphone",
            "group": "audio",
            "desc": "microphone=麦克风输入；loopback=系统正在播放的声音"
                    "（Windows WASAPI 回环，来源/设备改动后重新开关模块生效）",
        },
        "microphone": {
            "label": "麦克风设备", "type": "str", "default": "",
            "group": "audio",
            "desc": "设备名由模块自动扫描枚举（联动页为下拉选择框）；"
                    "留空用系统默认录音设备",
        },
        "speaker": {
            "label": "播放设备（回环）", "type": "str", "default": "",
            "group": "audio",
            "desc": "loopback 来源时回环采集的播放设备（自动枚举）；"
                    "留空用系统默认播放设备",
        },
        "swap_channels": {
            "label": "左右声道交换", "type": "bool", "default": False,
            "group": "audio",
            "desc": "默认采集声道0(物理左)→左变量、声道1(物理右)→右变量；"
                    "开启后交换（现场左右接反时对调，设备侧绑定在事件流配置）",
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
from modules.sound_link.bridge import (DEFAULT_EVENT_CARDS, PARAM_DEFS,
                                       SoundBridge, SoundConfig)

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

    def on_load(self, ctx) -> None:
        self.ctx = ctx
        if "events" not in ctx.settings:
            ctx.settings["events"] = [
                {"name": card["name"], "trigger": card["trigger"],
                 "arg": card["arg"],
                 "actions": [dict(a) for a in card["actions"]]}
                for card in DEFAULT_EVENT_CARDS]
            ctx.log("已播种默认事件流：每 100ms 把左右推流值推入 "
                    "in_pulse_a/b（联动页事件流可自行调整）")

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

    async def stop(self) -> None:
        if self.bridge is not None:
            await self.bridge.stop()

    def is_running(self) -> bool:
        return self.bridge is not None and bool(getattr(self.bridge,
                                                        "_running", False))
