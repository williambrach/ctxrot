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


# --- ctxrot ---
callback = CtxRotCallback(db_path="ctxrot-optimizer.db", store_content=True)

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


# --- Student program + tiny trainset ---
student = dspy.ChainOfThought("question -> answer")

trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
    dspy.Example(question="What is the capital of Brazil?", answer="Brasília").with_inputs("question"),
    dspy.Example(question="What is the capital of Australia?", answer="Canberra").with_inputs("question"),
    dspy.Example(question="What is the capital of Egypt?", answer="Cairo").with_inputs("question"),
]


def exact_match(example: dspy.Example, pred: dspy.Prediction, trace: object = None) -> bool:
    """Case-insensitive substring match on the gold answer."""
    return example.answer.lower() in (pred.answer or "").lower()


# --- Compile ---
# Each student invocation during compile() opens a fresh ctxrot session —
# so this run produces multiple short sessions, not one grouped session.
# Open the TUI to see all of them.
optimizer = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=2)
compiled = optimizer.compile(student, trainset=trainset)

# Smoke test the compiled program on a held-out question.
result = compiled(question="What is the capital of Canada?")

print(result.answer)
print(f"\nLast session: {callback.session_id}")
print("View dashboard:  ctxrot --db ctxrot.db")
