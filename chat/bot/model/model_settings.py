from bot.model.mistral import MistralSettings
from bot.model.stablelm_zephyr import StableLMZephyrSettings
from bot.model.zephyr import ZephyrSettings

SUPPORTED_MODELS = {
    "zephyr": ZephyrSettings,
    "mistral": MistralSettings,
    "stablelm-zephyr": StableLMZephyrSettings,
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_setting(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
