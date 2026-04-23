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

# --- Search tool (free ColBERTv2 endpoint, no API key needed) ---
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")


def search(query: str) -> str:
    """Search Wikipedia for relevant passages."""
    results = colbert(query, k=3)
    return "\n\n".join(r["long_text"] for r in results)  # ty: ignore[invalid-argument-type]


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

# --- ReAct agent ---
react = dspy.ReAct(
    "question -> answer",  # ty:ignore[invalid-argument-type]
    tools=[search],
    max_iters=5,
)

result = react(
    question="Which of the two founders of Apple Inc was older, and by how many years?",
)

print(result.answer)
print(f"\nSession: {callback.session_id}")
print("View dashboard:  ctxrot --db ctxrot.db")
