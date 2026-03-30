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


class ResearchSummary(dspy.Module):
    """Three-step pipeline: outline, key findings, final summary."""

    def __init__(self):
        super().__init__()
        self.outline = dspy.ChainOfThought("topic -> outline")
        self.findings = dspy.ChainOfThought("topic, outline -> key_findings")
        self.synthesize = dspy.Predict("topic, outline, key_findings -> summary")

    def forward(self, topic):
        step1 = self.outline(topic=topic)
        step2 = self.findings(topic=topic, outline=step1.outline)
        step3 = self.synthesize(
            topic=topic,
            outline=step1.outline,
            key_findings=step2.key_findings,
        )
        return step3


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

# --- Run ---
module = ResearchSummary()
result = module(topic="The impact of transformer architecture on NLP")

print(result.summary)
print(f"\nSession: {callback.session_id}")
print("View dashboard:  ctxrot --db ctxrot.db")
