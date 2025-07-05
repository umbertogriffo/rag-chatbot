from enum import Enum

from bot.model.settings.deep_seek import DeepSeekR1SevenSettings
from bot.model.settings.llama import Llama31Settings, Llama31ToolSettings, Llama32OneSettings, Llama32ThreeSettings
from bot.model.settings.openchat import OpenChat35Settings, OpenChat36Settings
from bot.model.settings.phi import Phi35Settings
from bot.model.settings.qwen import Qwen25ThreeMathReasoningSettings, Qwen25ThreeSettings
from bot.model.settings.stablelm_zephyr import StableLMZephyrSettings
from bot.model.settings.starling import StarlingSettings


class Model(Enum):
    STABLELM_ZEPHYR = "stablelm-zephyr"
    OPENCHAT_3_5 = "openchat-3.5"
    OPENCHAT_3_6 = "openchat-3.6"
    STARLING = "starling"
    PHI_3_5 = "phi-3.5"
    LLAMA_3_1 = "llama-3.1"
    LLAMA_3_1_tool = "llama-3.1-tool"
    LLAMA_3_2_one = "llama-3.2:1b"
    LLAMA_3_2_three = "llama-3.2"
    QWEN_2_5_THREE = "qwen-2.5:3b"
    QWEN_2_5_THREE_MATH_REASONING = "qwen-2.5:3b-math-reasoning"
    DEEP_SEEK_R1_SEVEN = "deep-seek-r1:7b"


SUPPORTED_MODELS = {
    Model.STABLELM_ZEPHYR.value: StableLMZephyrSettings,
    Model.OPENCHAT_3_5.value: OpenChat35Settings,
    Model.OPENCHAT_3_6.value: OpenChat36Settings,
    Model.STARLING.value: StarlingSettings,
    Model.PHI_3_5.value: Phi35Settings,
    Model.LLAMA_3_1.value: Llama31Settings,
    Model.LLAMA_3_1_tool.value: Llama31ToolSettings,
    Model.LLAMA_3_2_one.value: Llama32OneSettings,
    Model.LLAMA_3_2_three.value: Llama32ThreeSettings,
    Model.QWEN_2_5_THREE.value: Qwen25ThreeSettings,
    Model.QWEN_2_5_THREE_MATH_REASONING.value: Qwen25ThreeMathReasoningSettings,
    Model.DEEP_SEEK_R1_SEVEN.value: DeepSeekR1SevenSettings,
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_settings(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
