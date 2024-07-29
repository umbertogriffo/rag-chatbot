from enum import Enum

from bot.model.settings.dolphin import DolphinSettings
from bot.model.settings.llama_3 import Llama31Settings
from bot.model.settings.mistral import MistralSettings
from bot.model.settings.openchat import OpenChat35Settings, OpenChat36Settings
from bot.model.settings.phi_3 import PhiThreeSettings
from bot.model.settings.stablelm_zephyr import StableLMZephyrSettings
from bot.model.settings.starling import StarlingSettings
from bot.model.settings.zephyr import ZephyrSettings


class ModelType(Enum):
    ZEPHYR = "zephyr"
    MISTRAL = "mistral"
    DOLPHIN = "dolphin"
    STABLELM_ZEPHYR = "stablelm-zephyr"
    OPENCHAT_3_5 = "openchat-3.5"
    OPENCHAT_3_6 = "openchat-3.6"
    STARLING = "starling"
    PHI_3 = "phi-3"
    LLAMA_3_1 = "llama-3.1"


SUPPORTED_MODELS = {
    ModelType.ZEPHYR.value: ZephyrSettings,
    ModelType.MISTRAL.value: MistralSettings,
    ModelType.DOLPHIN.value: DolphinSettings,
    ModelType.STABLELM_ZEPHYR.value: StableLMZephyrSettings,
    ModelType.OPENCHAT_3_5.value: OpenChat35Settings,
    ModelType.OPENCHAT_3_6.value: OpenChat36Settings,
    ModelType.STARLING.value: StarlingSettings,
    ModelType.PHI_3.value: PhiThreeSettings,
    ModelType.LLAMA_3_1.value: Llama31Settings,
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_setting(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
