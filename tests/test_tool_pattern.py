import json
import os
from types import SimpleNamespace

import pytest

from agentic_patterns.tool_pattern.tool import get_fn_signature
from agentic_patterns.tool_pattern.tool import tool
from agentic_patterns.tool_pattern.tool import validate_arguments
from agentic_patterns.tool_pattern.tool_agent import ToolAgent
from agentic_patterns.utils.extraction import extract_tag_content


def get_current_weather(location: str, unit: str):
    """
    Get the current weather in a given location

    location (str): The city and state, e.g. Madrid, Barcelona
    """
    temp = 25 if location == "Madrid" else 58
    return json.dumps({"temperature": temp, "unit": unit})


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# --- tool.py ---------------------------------------------------------------


def test_get_fn_signature_builds_schema_without_return():
    sig = get_fn_signature(add)
    assert sig["name"] == "add"
    assert sig["description"] == "Add two numbers."
    assert sig["parameters"]["properties"] == {
        "a": {"type": "int"},
        "b": {"type": "int"},
    }


def test_get_fn_signature_cleans_docstring_indentation():
    sig = get_fn_signature(get_current_weather)
    assert sig["description"].startswith("Get the current weather")
    assert "\n    " not in sig["description"]


def test_tool_decorator_returns_tool():
    t = tool(add)
    assert t.name == "add"
    assert json.loads(t.fn_signature)["name"] == "add"
    assert t.run(a=2, b=3) == 5


def test_validate_arguments_casts_types():
    call = {"name": "add", "arguments": {"a": "2", "b": 3.0}}
    out = validate_arguments(call, get_fn_signature(add))
    assert out["arguments"] == {"a": 2, "b": 3}
    assert all(isinstance(v, int) for v in out["arguments"].values())


def test_validate_arguments_ignores_unknown_args():
    call = {"name": "add", "arguments": {"a": 1, "b": 2, "extra": "x"}}
    out = validate_arguments(call, get_fn_signature(add))
    assert out["arguments"]["extra"] == "x"


def test_extract_tag_content_finds_multiple_calls():
    text = '<tool_call>{"a": 1}</tool_call> hi <tool_call>\n{"b": 2}\n</tool_call>'
    res = extract_tag_content(text, "tool_call")
    assert res.found
    assert res.content == ['{"a": 1}', '{"b": 2}']


# --- tool_agent.py (LLM mocked) --------------------------------------------


class FakeClient:
    """Mimics Groq: returns queued responses and records the messages sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages, model, **kwargs):
        self.calls.append(list(messages))
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def make_agent(responses, tools):
    agent = ToolAgent.__new__(ToolAgent)
    agent.client = FakeClient(responses)
    agent.model = "fake"
    agent.max_tokens = 100
    agent.tools = tools
    agent.tools_dict = {t.name: t for t in tools}
    return agent


def test_agent_runs_tool_and_sends_observation():
    weather = tool(get_current_weather)
    agent = make_agent(
        [
            '<tool_call>{"name": "get_current_weather", "arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>',
            "It's 25 C in Madrid.",
        ],
        [weather],
    )
    assert agent.run("Weather in Madrid?") == "It's 25 C in Madrid."

    final_msgs = agent.client.calls[1]
    observation = final_msgs[-1]["content"]
    assert observation.startswith("Observation: ")  # no stray f"..." wrapper
    assert '"temperature": 25' in observation


def test_agent_without_tool_call_skips_tools():
    agent = make_agent(["no tools needed", "I'm an assistant."], [tool(add)])
    assert agent.run("Tell me your name") == "I'm an assistant."
    assert len(agent.client.calls[1]) == 1  # only the user message


def test_agent_handles_missing_id_and_unknown_tool():
    agent = make_agent(
        [
            '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>'
            '<tool_call>{"name": "nope", "arguments": {}}</tool_call>',
            "3",
        ],
        [tool(add)],
    )
    assert agent.run("1+2") == "3"
    assert "{'add': 3}" in agent.client.calls[1][-1]["content"]


def test_multiple_tool_signatures_are_separated():
    agent = make_agent([], [tool(add), tool(get_current_weather)])
    lines = agent.add_tool_signatures().splitlines()
    assert [json.loads(line)["name"] for line in lines] == ["add", "get_current_weather"]


# --- live Groq call ----------------------------------------------------------


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_live_default_model_emits_tool_call():
    agent = ToolAgent(tools=[tool(get_current_weather)])
    called = []
    agent.tools_dict["get_current_weather"].fn = lambda **kw: called.append(kw) or "25"
    out = agent.run("What's the current temperature in Madrid, in Celsius?")
    assert called and "Madrid" in called[0]["location"]
    assert out.strip()
