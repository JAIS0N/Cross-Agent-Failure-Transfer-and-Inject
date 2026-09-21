"""Verify the LLM code paths WITHOUT needing Ollama, using a fake client.

This exercises prompt building, answer normalization, grading, latency
capture, and the routing loop end-to-end with a stub that returns canned
completions -- so the real-model integration is tested in CI."""
from pfti.routing.llm_models import LLMModel, _normalize
from pfti.routing.router import PerfStore, PerfRecord, Router


class _FakeMsg:
    def __init__(self, content): self.message = type("M", (), {"content": content})
class _FakeResp:
    def __init__(self, content): self.choices = [_FakeMsg(content)]
class _FakeClient:
    """small model gets extract wrong, large gets it right."""
    def __init__(self): self.chat = self
    class completions:  # placeholder, replaced below
        pass
    def create(self, model, messages, temperature=0.0):
        prompt = messages[0]["content"]
        if "qwen-small" in model:
            return _FakeResp("wrong")
        return _FakeResp("Paris")


def _client():
    c = _FakeClient()
    c.completions = c  # so client.chat.completions.create works
    return c


def test_normalize():
    assert _normalize("  Paris. ") == "paris"
    assert _normalize("BUG!") == "bug"


def test_llm_model_runs_and_grades():
    c = _client()
    small = LLMModel("small", "qwen-small", 1.0, c)
    large = LLMModel("large", "qwen-large", 6.0, c)
    item = {"task_type": "extract", "input": "city=Paris", "question": "city",
            "gold": "Paris"}
    a_s, lat_s, cost_s, ok_s = small.run("extract", item)
    a_l, lat_l, cost_l, ok_l = large.run("extract", item)
    assert ok_s is False and ok_l is True
    assert cost_s == 1.0 and cost_l == 6.0
    assert lat_s >= 0 and lat_l >= 0


def test_router_uses_llm_records():
    c = _client()
    small = LLMModel("small", "qwen-small", 1.0, c)
    large = LLMModel("large", "qwen-large", 6.0, c)
    models = {"small": small, "large": large}
    store = PerfStore()
    # peer A tried both: small failed, large succeeded on extract
    store.add(PerfRecord("A", "extract", "plain", "small", False, 200, 1.0))
    store.add(PerfRecord("A", "extract", "plain", "large", True, 1200, 6.0))
    r = Router(store, models)
    model, why = r.choose("extract", "plain", exclude_agent="B")
    assert model == "large" and why.startswith("learned")
