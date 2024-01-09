from enum import Enum

from bot.model.mistral import MistralSettings
from bot.model.neural_marcoro import NeuralMarcoroSettings
from bot.model.openchat import OpenChatSettings
from bot.model.stablelm_zephyr import StableLMZephyrSettings
from bot.model.zephyr import ZephyrSettings


class ModelType(Enum):
    ZEPHYR = "zephyr"
    MISTRAL = "mistral"
    STABLELM_ZEPHYR = "stablelm-zephyr"
    OPENCHAT = "openchat"
    NEURAL_MARCORO = "neural-marcoro"


SUPPORTED_MODELS = {
    ModelType.ZEPHYR.value: ZephyrSettings,
    ModelType.MISTRAL.value: MistralSettings,
    ModelType.STABLELM_ZEPHYR.value: StableLMZephyrSettings,
    ModelType.OPENCHAT.value: OpenChatSettings,
    ModelType.NEURAL_MARCORO.value: NeuralMarcoroSettings,
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_setting(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
