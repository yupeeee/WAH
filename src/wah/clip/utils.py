from huggingface_hub import list_models

__all__ = [
    "available_models",
]


def available_models():
    return [model.modelId for model in list_models(filter="clip", author="openai")]
