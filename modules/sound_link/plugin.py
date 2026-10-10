META = {
    "id": "sound_link",
    "name": "音频联动",
    "version": "0.5.3",
    "description": "纯输入采集：麦克风与系统声音（WASAPI 回环）双路同时监听，"
                   "每路各输出左右响度 / 左右频率八个只读变量。模块只采集与登记"
                   "变量、绝不直接驱动设备；脉冲流推入与设备动作一律在「事件流」页"
                   "用写入卡（如范围映射卡把频率变量映射到核心「外部脉冲流频率」）"
                   "完成，由用户自行连线。",
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
        "loop_enabled": {
            "label": "监听系统声音", "type": "bool", "default": False,
            "group": "audio",
            "desc": "Windows WASAPI 回环采集正在播放的声音；与麦克风互不影响，"
                    "两路各输出八个变量中的四个（改动后重新开关模块生效）",
        },
        "speaker": {
            "label": "播放设备（回环）", "type": "str", "default": "",
            "group": "audio",
            "desc": "回环采集的播放设备（自动枚举）；留空用系统默认播放设备",
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

    def link_params(self) -> list[dict]:
        """八个采集变量：显式声明为只读（dir=in）、浮点（type=Float）。"""
        return [{"name": name, "label": str(item.get("label") or ""),
                 "dir": "in", "type": "Float",
                 "desc": str(item.get("desc") or "")}
                for name, item in PARAM_DEFS.items()]

    def temp_specs(self) -> list[dict]:
        """采集值由本模块每拍维护：只读，回传方向没有意义。"""
        return [{"key": name, "label": str(item.get("label") or ""),
                 "dir": "in", "type": "Float",
                 "desc": str(item.get("desc") or "")}
                for name, item in PARAM_DEFS.items()]

    def on_load(self, ctx) -> None:
        self.ctx = ctx
        if migrate_settings(ctx.settings, ctx.log) and hasattr(ctx.settings, "save"):
            ctx.settings.save()
        ctx.settings.pop("events", None)   # 清掉历史遗留的事件卡播种字段

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
