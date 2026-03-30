import math
import os

import dspy
from dotenv import load_dotenv

from ctxrot import CtxRotCallback

dspy.cache.enable_disk_cache = False
dspy.cache.enable_memory_cache = False

load_dotenv()

MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")


# --- Tools (pure Python, no external services) ---
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    allowed = set("0123456789+-*/().eE ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions allowed"
    return str(eval(expression))  # noqa: S307


def lookup_data(key: str) -> str:
    """Look up a scientific/demographic fact."""
    data = {
        "world_population_2024": "8.1 billion (8_100_000_000)",
        "earth_circumference_km": "40_075",
        "earth_radius_km": "6_371",
        "speed_of_light_ms": "299_792_458",
        "pi": str(math.pi),
    }
    normalized_key = key.lower()
    available_keys = ", ".join(data)
    fallback = f"No data found for '{key}'. Available keys: {available_keys}"
    return data.get(normalized_key, fallback)


# --- ctxrot ---
callback = CtxRotCallback(db_path="ctxrot.db", store_content=True)

lm_kwargs = {}
if API_KEY:
    lm_kwargs["api_key"] = API_KEY
if API_BASE:
    lm_kwargs["api_base"] = API_BASE

dspy.configure(
    lm=dspy.LM(MODEL, **lm_kwargs),
    callbacks=[callback],
    cache=False,  # disable caching to see all calls in ctxrot
)

# --- RLM agent ---
rlm = dspy.RLM(
    signature="question -> answer",
    tools=[calculate, lookup_data],
    max_iterations=10,
    max_llm_calls=5,
)

result = rlm(
    question="If you lined up all humans on Earth shoulder to shoulder "
    "(0.5m each), how many times would the line wrap around "
    "the Earth at the equator?",
)

print(result.answer)
print(f"\nSession: {callback.session_id}")
print("View dashboard:  ctxrot --db ctxrot-rlm.db")
