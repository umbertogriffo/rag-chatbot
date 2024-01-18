from enum import Enum

from bot.model.dolphin import DolphinSettings
from bot.model.mistral import MistralSettings
from bot.model.neural_beagle import NeuralBeagleSettings
from bot.model.openchat import OpenChatSettings
from bot.model.stablelm_zephyr import StableLMZephyrSettings
from bot.model.zephyr import ZephyrSettings


class ModelType(Enum):
    ZEPHYR = "zephyr"
    MISTRAL = "mistral"
    DOLPHIN = "dolphin"
    STABLELM_ZEPHYR = "stablelm-zephyr"
    OPENCHAT = "openchat"
    NEURAL_BEAGLE = "neural-beagle"


SUPPORTED_MODELS = {
    ModelType.ZEPHYR.value: ZephyrSettings,
    ModelType.MISTRAL.value: MistralSettings,
    ModelType.DOLPHIN.value: DolphinSettings,
    ModelType.STABLELM_ZEPHYR.value: StableLMZephyrSettings,
    ModelType.OPENCHAT.value: OpenChatSettings,
    ModelType.NEURAL_BEAGLE.value: NeuralBeagleSettings,
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_setting(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
