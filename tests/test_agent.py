"""Tests for the agent loop with a mock LLM client."""

from __future__ import annotations

from typing import Any

import pytest

from agent.context.discovery import ProjectContext
from agent.core.agent import Agent
from agent.core.llm import LLMResponse, ToolCall


class MockLLMClient:
    """LLM client that returns scripted responses in sequence."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.send_calls: list[dict[str, Any]] = []

    def send(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        *,
        stream_text: bool = False,
    ) -> LLMResponse:
        self.send_calls.append(
            {"messages": messages, "system": system, "tools": tools}
        )
        if self._call_index >= len(self._responses):
            return LLMResponse(text="No more scripted responses.", stop_reason="end_turn")
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp


@pytest.fixture()
def context(project_root):
    return ProjectContext(project_root)


class TestAgentBasicFlow:
    def test_immediate_text_response(self, warehouse, context):
        """Agent returns immediately if LLM gives text without tool calls."""
        mock = MockLLMClient([
            LLMResponse(text="The answer is 42.", tool_calls=[], stop_reason="end_turn"),
        ])
        agent = Agent(mock, warehouse, context)
        result = agent.run("What is the meaning of life?")
        assert result == "The answer is 42."

    def test_single_tool_call(self, warehouse, context):
        """Agent handles one tool call then a final text response."""
        mock = MockLLMClient([
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="tc1", name="list_schemas", input={})
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="The warehouse has schemas: raw, staging, marts.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ])
        agent = Agent(mock, warehouse, context)
        result = agent.run("What schemas exist?")
        assert "raw" in result or "schemas" in result.lower()

    def test_multi_step_tool_calls(self, warehouse, context):
        """Agent can chain multiple tool calls before answering."""
        mock = MockLLMClient([
            # Step 1: list schemas
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="list_schemas", input={})],
                stop_reason="tool_use",
            ),
            # Step 2: list tables in marts
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc2", name="list_tables", input={"schema": "marts"})],
                stop_reason="tool_use",
            ),
            # Step 3: run a query
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc3",
                        name="run_query",
                        input={"sql": "SELECT COUNT(*) as cnt FROM marts.fct_orders"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Final answer
            LLMResponse(
                text="There are 8 orders in the database.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ])
        agent = Agent(mock, warehouse, context)
        result = agent.run("How many orders are there?")
        assert "8" in result
        assert mock._call_index == 4

    def test_sql_tracked_in_last_sql(self, warehouse, context):
        """Agent tracks the last SQL query executed."""
        mock = MockLLMClient([
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="tc1", name="run_query", input={"sql": "SELECT 1"})
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Done.", tool_calls=[], stop_reason="end_turn"),
        ])
        agent = Agent(mock, warehouse, context)
        agent.run("test")
        assert agent.last_sql == "SELECT 1"


class TestAgentErrorHandling:
    def test_tool_error_is_sent_back(self, warehouse, context):
        """If a tool raises an exception, the error is sent back to the LLM."""
        mock = MockLLMClient([
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="run_query",
                        input={"sql": "SELECT * FROM nonexistent"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                text="That table doesn't exist. Let me try something else.",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ])
        agent = Agent(mock, warehouse, context)
        result = agent.run("Query nonexistent table")
        assert "doesn't exist" in result or "something else" in result

        # Verify the error was sent as a tool_result with is_error=True
        second_call = mock.send_calls[1]
        last_msg = second_call["messages"][-1]
        assert last_msg["role"] == "user"
        tool_result = last_msg["content"][0]
        assert tool_result["is_error"] is True
        assert "Error" in tool_result["content"]

    def test_max_iterations_cap(self, warehouse, context):
        """Agent stops after max_iterations even if LLM keeps requesting tools."""
        # Create an infinite tool-call loop
        infinite_responses = [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id=f"tc{i}", name="list_schemas", input={})],
                stop_reason="tool_use",
            )
            for i in range(20)
        ]
        mock = MockLLMClient(infinite_responses)
        agent = Agent(mock, warehouse, context, max_iterations=3)
        result = agent.run("Loop forever")
        assert "maximum number of iterations" in result
        assert mock._call_index == 3


class TestAgentMessageBuilding:
    def test_system_prompt_included(self, warehouse, context):
        """Agent sends system prompt to LLM."""
        mock = MockLLMClient([
            LLMResponse(text="ok", tool_calls=[], stop_reason="end_turn"),
        ])
        agent = Agent(mock, warehouse, context)
        agent.run("test")
        assert "data warehouse" in mock.send_calls[0]["system"].lower()

    def test_tools_included(self, warehouse, context):
        """Agent sends tool definitions to LLM."""
        mock = MockLLMClient([
            LLMResponse(text="ok", tool_calls=[], stop_reason="end_turn"),
        ])
        agent = Agent(mock, warehouse, context)
        agent.run("test")
        tools = mock.send_calls[0]["tools"]
        tool_names = {t["name"] for t in tools}
        assert "list_schemas" in tool_names
        assert "run_query" in tool_names

    def test_tool_results_in_messages(self, warehouse, context):
        """Tool results are correctly appended to message history."""
        mock = MockLLMClient([
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="list_schemas", input={})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="ok", tool_calls=[], stop_reason="end_turn"),
        ])
        agent = Agent(mock, warehouse, context)
        agent.run("test")

        # Second call should have: user question, assistant tool_use, user tool_result
        messages = mock.send_calls[1]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"


class TestAgentConversationHistory:
    def test_second_question_includes_history(self, warehouse, context):
        """Second question includes prior Q&A in messages."""
        mock = MockLLMClient([
            LLMResponse(text="Answer 1.", tool_calls=[], stop_reason="end_turn"),
            LLMResponse(text="Answer 2.", tool_calls=[], stop_reason="end_turn"),
        ])
        agent = Agent(mock, warehouse, context)
        agent.run("Question 1")
        agent.run("Question 2")

        # Second call should have history: Q1, A1, then Q2
        messages = mock.send_calls[1]["messages"]
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Question 1"}
        assert messages[1] == {"role": "assistant", "content": "Answer 1."}
        assert messages[2] == {"role": "user", "content": "Question 2"}

    def test_schema_cache_injected_after_first_question(self, warehouse, context):
        """After first question explores schema, system prompt includes cached schema."""
        mock = MockLLMClient([
            # First question: explore schema
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="list_schemas", input={})],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Found schemas.", tool_calls=[], stop_reason="end_turn"),
            # Second question: should see schema in system prompt
            LLMResponse(text="Answer 2.", tool_calls=[], stop_reason="end_turn"),
        ])
        agent = Agent(mock, warehouse, context)
        agent.run("What schemas?")
        agent.run("Follow up")

        # Third LLM call (second question) should have schema context
        system_prompt = mock.send_calls[2]["system"]
        assert "raw" in system_prompt
        assert "staging" in system_prompt
        assert "marts" in system_prompt
        assert "skip introspection" in system_prompt.lower()

    def test_conversation_turn_recorded(self, warehouse, context):
        """Agent records turns in conversation manager."""
        mock = MockLLMClient([
            LLMResponse(text="42 orders.", tool_calls=[], stop_reason="end_turn"),
        ])
        agent = Agent(mock, warehouse, context)
        agent.run("How many orders?")

        assert agent.conversation.turn_count == 1
        assert agent.conversation.recent_turns[0]["question"] == "How many orders?"
        assert agent.conversation.recent_turns[0]["answer"] == "42 orders."
