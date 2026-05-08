from .qwen import QwenAdapter

ADAPTER_REGISTRY = {
    "qwen": QwenAdapter,
}

# Threshold strategy per adapter type
THRESHOLD_STRATEGY = {
    "qwen": "fixed",
}


def build_adapter(model_type: str, model_key: str, model_path: str, device: str):
    if model_type not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[model_type](model_key, model_path, device)


__all__ = ["ADAPTER_REGISTRY", "THRESHOLD_STRATEGY", "build_adapter", "QwenAdapter"]
