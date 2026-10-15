"""LAB bridge to local OpenAI Chat Completions (official adapter uses Responses)."""

from typing import Any
import os

from lab_core.harness.adapters.base import ModelAdapter, ModelResponse, ToolCall
from openai import OpenAI


class LocalChatAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        client: Any = None,
    ):
        super().__init__(model, temperature, reasoning_effort)
        self.client = client or OpenAI(
            base_url=os.environ.get("LAB_LOCAL_BASE_URL", "http://127.0.0.1:8001/v1"),
            api_key=os.environ.get("LAB_LOCAL_API_KEY", "local-only"),
            timeout=600,
        )
        self.max_tokens = int(os.environ.get("LAB_MAX_OUTPUT_TOKENS", "4096"))

    def chat(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[{"type": "function", "function": t} for t in tools],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not response.choices:
            raise ValueError("Empty local model response")
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        message.pop("reasoning_content", None)
        calls = [
            ToolCall(id=t.id, name=t.function.name, arguments=t.function.arguments)
            for t in (choice.message.tool_calls or [])
        ]
        return ModelResponse(
            message=message,
            tool_calls=calls,
            text=choice.message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=choice.finish_reason,
        )

    def make_tool_result_messages(self, results: list[tuple[str, str]]) -> list[dict]:
        return [
            {"role": "tool", "tool_call_id": tid, "content": value}
            for tid, value in results
        ]

    def make_system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def make_user_message(self, content: str) -> dict:
        return {"role": "user", "content": content}
