import re

import litellm
from litellm import cost_calculator

_SEPARATORS = set("-.:_")


def _strip_ft_suffix(model: str) -> str:
    """Strip OpenAI fine-tune suffix: 'ft:gpt-4.1-mini:org:suffix:id' -> 'ft:gpt-4.1-mini'."""
    return re.sub(r"(:[^:]+){3}$", "", model)


def _find_base_model_info(model: str) -> dict | None:
    """Match a custom model name to a known base model by longest prefix.

    Only matches at separator boundaries to avoid false positives
    like 'gpt-4.10' matching 'gpt-4.1'.
    """
    for i in range(len(model) - 1, 0, -1):
        if model[i] not in _SEPARATORS:
            continue
        candidate = model[:i]
        info = litellm.model_cost.get(candidate)
        if info:
            return info
    return None


def get_model_info(model: str) -> dict | None:
    """Look up model info from litellm's pricing database.

    Resolution order:
    1. Exact match
    2. Strip provider prefix ('openai/gpt-4.1-mini' -> 'gpt-4.1-mini')
    3. Strip ft: org/suffix/id ('ft:gpt-4.1:org:suf:id' -> 'ft:gpt-4.1')
    4. Longest prefix match at separator boundary ('gpt-4.1-mini-fiit' -> 'gpt-4.1-mini')
    5. Normalize dots to hyphens and retry exact + prefix match
    """
    # 1. Exact match
    info = litellm.model_cost.get(model)
    if info:
        return info

    # 2. Strip provider prefix
    stripped = model.split("/", 1)[1] if "/" in model else model
    if stripped != model:
        info = litellm.model_cost.get(stripped)
        if info:
            return info

    # 3. Strip ft: fine-tune suffix (ft:base:org:suffix:id -> ft:base)
    ft_stripped = _strip_ft_suffix(stripped) if ":" in stripped else stripped
    if ft_stripped != stripped:
        info = litellm.model_cost.get(ft_stripped)
        if info:
            return info
        # Also try without the ft: prefix (ft:gpt-4.1-mini -> gpt-4.1-mini)
        if ft_stripped.startswith("ft:"):
            base = ft_stripped[3:]
            info = litellm.model_cost.get(base)
            if info:
                return info

    # 4. Longest prefix match
    info = _find_base_model_info(ft_stripped)
    if info:
        return info

    # 5. Normalize dots to hyphens and retry (claude-sonnet-4.6 -> claude-sonnet-4-6)
    normalized = ft_stripped.replace(".", "-")
    if normalized != ft_stripped:
        info = litellm.model_cost.get(normalized)
        if info:
            return info
        info = _find_base_model_info(normalized)
        if info:
            return info

    return None


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Calculate cost using litellm's cost_per_token. Returns None if model unknown."""
    try:
        prompt_cost, completion_cost = cost_calculator.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cache_read_tokens,
            cache_creation_input_tokens=cache_write_tokens,
        )
        return prompt_cost + completion_cost
    except Exception:
        return None


def is_model_known(model: str) -> bool:
    """Return True if we can resolve the model to a known entry."""
    return get_model_info(model) is not None


def get_context_window(model: str) -> int:
    """Return model's max input tokens. Defaults to 200,000 if unknown."""
    info = get_model_info(model)
    if info:
        return info.get("max_input_tokens") or info.get("max_tokens") or 200_000
    return 200_000
