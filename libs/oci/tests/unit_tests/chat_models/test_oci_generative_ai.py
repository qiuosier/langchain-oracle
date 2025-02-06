"""Test OCI Generative AI LLM service"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pytest import MonkeyPatch

from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI


class MockResponseDict(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
        return self[val]


class MockToolCall(dict):
    def __getattr__(self, val):  # type: ignore[no-untyped-def]
        return self[val]


@pytest.mark.requires("oci")
@pytest.mark.parametrize(
    "test_model_id", ["cohere.command-r-16k", "meta.llama-3-70b-instruct"]
)
def test_llm_chat(monkeypatch: MonkeyPatch, test_model_id: str) -> None:
    """Test valid chat call to OCI Generative AI LLM service."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id=test_model_id, client=oci_gen_ai_client)

    model_id = llm.model_id
    if model_id is None:
        raise ValueError("Model ID is required for OCI Generative AI LLM service.")

    provider = model_id.split(".")[0].lower()

    def mocked_response(*args):  # type: ignore[no-untyped-def]
        response_text = "Assistant chat reply."
        response = None
        if provider == "cohere":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "text": response_text,
                                    "finish_reason": "completed",
                                    "is_search_required": None,
                                    "search_queries": None,
                                    "citations": None,
                                    "documents": None,
                                    "tool_calls": None,
                                }
                            ),
                            "model_id": "cohere.command-r-16k",
                            "model_version": "1.0.0",
                        }
                    ),
                    "request_id": "1234567890",
                    "headers": MockResponseDict(
                        {
                            "content-length": "123",
                        }
                    ),
                }
            )
        elif provider == "meta":
            response = MockResponseDict(
                {
                    "status": 200,
                    "data": MockResponseDict(
                        {
                            "chat_response": MockResponseDict(
                                {
                                    "choices": [
                                        MockResponseDict(
                                            {
                                                "message": MockResponseDict(
                                                    {
                                                        "content": [
                                                            MockResponseDict(
                                                                {
                                                                    "text": response_text,  # noqa: E501
                                                                }
                                                            )
                                                        ], 
                                                        "tool_calls": [
                                                            MockResponseDict(
                                                                {
                                                                    "type": "function",
                                                                    "id": "call_123",
                                                                    "function": {
                                                                        "name": "get_weather",
                                                                    }
                                                                }
                                                            )
                                                        ]
                                                    }
                                                ),
                                                "finish_reason": "completed",
                                            }
                                        )
                                    ],
                                    "time_created": "2024-09-01T00:00:00Z",
                                }
                            ),
                            "model_id": "meta.llama-3.1-70b-instruct",
                            "model_version": "1.0.0",
                        }
                    ),
                    "request_id": "1234567890",
                    "headers": MockResponseDict(
                        {
                            "content-length": "123",
                        }
                    ),
                }
            )
        return response

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    messages = [
        HumanMessage(content="User message"),
    ]

    expected = "Assistant chat reply."
    actual = llm.invoke(messages, temperature=0.2)
    assert actual.content == expected


@pytest.mark.requires("oci")
def test_meta_tool_calling(monkeypatch: MonkeyPatch) -> None:
    """Test tool calling with Meta models."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3-70b-instruct", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Verify tool choice is correctly set in request
        request = args[0]
        
        # Mock response with tool calls
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "choices": [
                                    MockResponseDict(
                                        {
                                            "message": MockResponseDict(
                                                {
                                                    "content": [
                                                        MockResponseDict(
                                                            {
                                                                "text": "Let me help you with that.",
                                                            }
                                                        )
                                                    ],
                                                    "tool_calls": [
                                                        MockResponseDict(
                                                            {
                                                                "type": "function",
                                                                "id": "call_456",
                                                                "function": {
                                                                    "name": "get_weather",
                                                                    "arguments": '{"location": "San Francisco"}',
                                                                },
                                                            }
                                                        )
                                                    ],
                                                }
                                            ),
                                            "finish_reason": "completed",
                                        }
                                    )
                                ],
                                "time_created": "2024-09-01T00:00:00Z",
                            }
                        ),
                        "model_id": "meta.llama-3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    # Define a simple weather tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather for {location}"

    messages = [HumanMessage(content="What's the weather like?")]

    # Test different tool choice options
    tool_choices = [
        "get_weather",  # Specific tool
        "auto",  # Auto mode
        "none",  # No tools
        True,  # Required
        False,  # None
        {"type": "function", "function": {"name": "get_weather"}},  # Dict format
    ]

    for tool_choice in tool_choices:
        response = llm.bind_tools(
            tools=[get_weather],
            tool_choice=tool_choice,
        ).invoke(messages)
        
        assert response.content == "Let me help you with that."
        if tool_choice not in ["none", False]:
            assert response.additional_kwargs.get("tool_calls") is not None
            tool_call = response.additional_kwargs["tool_calls"][0]
            assert tool_call["type"] == "function"
            assert tool_call["function"]["name"] == "get_weather"


@pytest.mark.requires("oci")
def test_cohere_tool_choice_validation(monkeypatch: MonkeyPatch) -> None:
    """Test that tool choice is not supported for Cohere models."""
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="cohere.command-r-16k", client=oci_gen_ai_client)

    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"Weather for {location}"

    messages = [HumanMessage(content="What's the weather like?")]

    # Test that tool choice raises ValueError
    with pytest.raises(ValueError, match="Tool choice is not supported for Cohere models"):
        llm.bind_tools(
            tools=[get_weather],
            tool_choice="auto",
        ).invoke(messages)

    # Mock response for the case without tool choice
    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "text": "Response without tool choice",
                                "finish_reason": "completed",
                                "is_search_required": None,
                                "search_queries": None,
                                "citations": None,
                                "documents": None,
                                "tool_calls": None,
                            }
                        ),
                        "model_id": "cohere.command-r-16k",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    # Test that tools without tool choice works
    response = llm.bind_tools(tools=[get_weather]).invoke(messages)
    assert response.content == "Response without tool choice"


@pytest.mark.requires("oci")
def test_meta_tool_conversion(monkeypatch: MonkeyPatch) -> None:
    """Test tool conversion for Meta models."""
    from pydantic import BaseModel, Field
    
    oci_gen_ai_client = MagicMock()
    llm = ChatOCIGenAI(model_id="meta.llama-3-70b-instruct", client=oci_gen_ai_client)

    def mocked_response(*args, **kwargs):  # type: ignore[no-untyped-def]
        
        return MockResponseDict(
            {
                "status": 200,
                "data": MockResponseDict(
                    {
                        "chat_response": MockResponseDict(
                            {
                                "choices": [
                                    MockResponseDict(
                                        {
                                            "message": MockResponseDict(
                                                {
                                                    "content": [
                                                        MockResponseDict(
                                                            {"text": "Response"}
                                                        )
                                                    ]
                                                }
                                            ),
                                            "finish_reason": "completed",
                                        }
                                    )
                                ],
                                "time_created": "2024-09-01T00:00:00Z",
                            }
                        ),
                        "model_id": "meta.llama-3-70b-instruct",
                        "model_version": "1.0.0",
                    }
                ),
                "request_id": "1234567890",
                "headers": MockResponseDict({"content-length": "123"}),
            }
        )

    monkeypatch.setattr(llm.client, "chat", mocked_response)

    # Test function tool
    def function_tool(x: int) -> int:
        """A simple function tool.
        
        Args:
            x: Input number
        """
        return x + 1

    # Test pydantic tool
    class PydanticTool(BaseModel):
        """A simple pydantic tool."""
        x: int = Field(description="Input number")
        y: str = Field(description="Input string")


    messages = [HumanMessage(content="Test message")]

    # Test that all tool types can be bound and used
    response = llm.bind_tools(
        tools=[function_tool, PydanticTool],
    ).invoke(messages)

    assert response.content == "Response"
