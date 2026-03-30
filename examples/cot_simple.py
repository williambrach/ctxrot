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

# --- ctxrot: just these 2 lines ---
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

# --- your DSPy code, unchanged ---
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What are the three laws of thermodynamics?")

print(result.answer)
print(f"\nSession: {callback.session_id}")
print("View dashboard:  ctxrot --db ctxrot.db")
