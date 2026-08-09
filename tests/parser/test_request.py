"""Tests for the request parser module."""

import json
import logging

import pytest

from llm_flow_viewer.parser.request import parse_request


# ---------------------------------------------------------------------------
# Sample request bodies
# ---------------------------------------------------------------------------

SAMPLE_REQUEST_BODY = {
    "model": "deepseek-v4-flash",
    "max_tokens": 131072,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello, how are you?"},
                {"type": "text", "text": "Here's some context."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I am fine, thank you!"},
            ],
        },
    ],
    "tools": [
        {
            "name": "Read",
            "description": "Read files from the filesystem",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "Execute",
            "description": "Execute shell commands",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    ],
    "system": [
        {"type": "text", "text": "You are Droid, an AI assistant."},
        {"type": "text", "text": "You have access to tools."},
    ],
    "thinking": {"type": "enabled"},
    "stream": True,
}


# ===================================================================
# Basic field extraction
# ===================================================================

class TestModelExtraction:
    """Tests for extracting the model field."""

    def test_extracts_model(self):
        """model field extracted correctly from request body."""
        content = json.dumps(SAMPLE_REQUEST_BODY).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert result.model == "deepseek-v4-flash"

    def test_model_defaults_to_empty_when_missing(self):
        """model field defaults to empty string when not present."""
        body = json.dumps({"max_tokens": 100}).encode("utf-8")
        result = parse_request(body)
        assert result is not None
        assert result.model == ""


class TestMaxTokensExtraction:
    """Tests for extracting the max_tokens field."""

    def test_extracts_max_tokens(self):
        """max_tokens field extracted correctly."""
        content = json.dumps(SAMPLE_REQUEST_BODY).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert result.max_tokens == 131072

    def test_max_tokens_defaults_to_zero_when_missing(self):
        """max_tokens defaults to 0 when not present."""
        body = json.dumps({"model": "test"}).encode("utf-8")
        result = parse_request(body)
        assert result is not None
        assert result.max_tokens == 0


class TestMessagesExtraction:
    """Tests for extracting the messages array."""

    def test_extracts_messages(self):
        """messages array extracted preserving role and content blocks."""
        content = json.dumps(SAMPLE_REQUEST_BODY).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "user"
        assert result.messages[1]["role"] == "assistant"

    def test_messages_content_blocks_preserved(self):
        """Content blocks within messages are preserved correctly."""
        content = json.dumps(SAMPLE_REQUEST_BODY).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        user_content = result.messages[0]["content"]
        assert isinstance(user_content, list)
        assert len(user_content) == 2
        assert user_content[0]["type"] == "text"
        assert user_content[0]["text"] == "Hello, how are you?"

    def test_messages_defaults_to_empty_list(self):
        """messages defaults to empty list when not present."""
        body = json.dumps({"model": "test"}).encode("utf-8")
        result = parse_request(body)
        assert result is not None
        assert result.messages == []

    def test_multi_turn_conversation(self):
        """Multi-turn conversation (system, user, assistant, user) preserves all messages in order."""
        multi_turn_body = {
            "model": "deepseek-v4-flash",
            "max_tokens": 4096,
            "system": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": "What is Python?",
                },
                {
                    "role": "assistant",
                    "content": "Python is a programming language.",
                },
                {
                    "role": "user",
                    "content": "Tell me more.",
                },
            ],
        }
        content = json.dumps(multi_turn_body).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert len(result.messages) == 3, f"Expected 3 messages, got {len(result.messages)}"
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "What is Python?"
        assert result.messages[1]["role"] == "assistant"
        assert result.messages[1]["content"] == "Python is a programming language."
        assert result.messages[2]["role"] == "user"
        assert result.messages[2]["content"] == "Tell me more."

    def test_messages_with_string_content(self):
        """Messages with plain string content (not list) are preserved correctly."""
        body = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }
        content = json.dumps(body).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert len(result.messages) == 2
        assert result.messages[0]["content"] == "Hello"
        assert result.messages[1]["content"] == "Hi there!"


class TestContentBlockTypes:
    """Tests for various content block types in messages."""

    def test_text_and_image_url_blocks(self):
        """Content blocks with text and image_url types are preserved."""
        body = {
            "model": "deepseek-v4-flash",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAA",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "Describe it in detail."},
                    ],
                },
            ],
        }
        content = json.dumps(body).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert len(result.messages) == 1
        blocks = result.messages[0]["content"]
        assert isinstance(blocks, list)
        assert len(blocks) == 3
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "What's in this image?"
        assert blocks[1]["type"] == "image_url"
        assert "url" in blocks[1]["image_url"]
        assert blocks[1]["image_url"]["detail"] == "high"
        assert blocks[2]["type"] == "text"
        assert blocks[2]["text"] == "Describe it in detail."

    def test_tool_use_in_assistant_message(self):
        """Assistant messages with tool_use content blocks preserve name, id, and input."""
        body = {
            "model": "deepseek-v4-flash",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": "Read the file foo.py",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll read that file.",
                        },
                        {
                            "type": "tool_use",
                            "id": "call_00_abc123",
                            "name": "Read",
                            "input": {
                                "file_path": "foo.py",
                            },
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_00_abc123",
                            "content": "def hello():\n    print('world')\n",
                        },
                    ],
                },
            ],
        }
        content = json.dumps(body).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert len(result.messages) == 3

        # Check tool_use block in assistant message
        assistant_content = result.messages[1]["content"]
        assert isinstance(assistant_content, list)
        tool_use_block = assistant_content[1]
        assert tool_use_block["type"] == "tool_use"
        assert tool_use_block["id"] == "call_00_abc123"
        assert tool_use_block["name"] == "Read"
        assert tool_use_block["input"] == {"file_path": "foo.py"}

        # Check tool_result block in user message
        user_content = result.messages[2]["content"]
        assert isinstance(user_content, list)
        tool_result_block = user_content[0]
        assert tool_result_block["type"] == "tool_result"
        assert tool_result_block["tool_use_id"] == "call_00_abc123"
        assert "def hello()" in tool_result_block["content"]


class TestToolsExtraction:
    """Tests for extracting the tools definitions."""

    def test_extracts_tools(self):
        """tools definitions extracted with name, description, input_schema."""
        content = json.dumps(SAMPLE_REQUEST_BODY).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert len(result.tools) == 2
        assert result.tools[0]["name"] == "Read"
        assert result.tools[0]["description"] == "Read files from the filesystem"
        assert "input_schema" in result.tools[0]

    def test_tools_defaults_to_empty_list(self):
        """tools defaults to empty list when not present."""
        body = json.dumps({"model": "test"}).encode("utf-8")
        result = parse_request(body)
        assert result is not None
        assert result.tools == []


class TestSystemExtraction:
    """Tests for extracting system prompts."""

    def test_extracts_system(self):
        """system prompts extracted as list of dicts."""
        content = json.dumps(SAMPLE_REQUEST_BODY).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert len(result.system) == 2
        assert result.system[0]["type"] == "text"
        assert "Droid" in result.system[0]["text"]

    def test_system_defaults_to_empty_list(self):
        """system defaults to empty list when not present."""
        body = json.dumps({"model": "test"}).encode("utf-8")
        result = parse_request(body)
        assert result is not None
        assert result.system == []


class TestThinkingExtraction:
    """Tests for extracting thinking configuration."""

    def test_extracts_thinking(self):
        """thinking config extracted as dict."""
        content = json.dumps(SAMPLE_REQUEST_BODY).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert result.thinking == {"type": "enabled"}

    def test_thinking_defaults_to_none(self):
        """thinking defaults to None when not present."""
        body = json.dumps({"model": "test"}).encode("utf-8")
        result = parse_request(body)
        assert result is not None
        assert result.thinking is None


class TestStreamExtraction:
    """Tests for extracting the stream flag."""

    def test_extracts_stream_true(self):
        """stream flag extracted as True."""
        content = json.dumps(SAMPLE_REQUEST_BODY).encode("utf-8")
        result = parse_request(content)
        assert result is not None
        assert result.stream is True

    def test_stream_defaults_to_false(self):
        """stream defaults to False when not present."""
        body = json.dumps({"model": "test"}).encode("utf-8")
        result = parse_request(body)
        assert result is not None
        assert result.stream is False


# ===================================================================
# Error handling
# ===================================================================

class TestErrorHandling:
    """Tests for graceful error handling."""

    def test_handles_malformed_json(self, caplog):
        """Malformed JSON returns None and logs a warning."""
        bad_content = b"this is not valid json"
        caplog.set_level(logging.WARNING)
        result = parse_request(bad_content)
        assert result is None

    def test_handles_empty_bytes(self):
        """Empty bytes returns None."""
        result = parse_request(b"")
        assert result is None

    def test_handles_none_content(self):
        """None content returns None."""
        result = parse_request(None)
        assert result is None


# ===================================================================
# Output config extraction
# ===================================================================

class TestOutputConfigExtraction:
    """Tests for extracting output_config."""

    def test_extracts_output_config(self):
        """output_config extracted when present."""
        body = {
            "model": "test",
            "output_config": {"effort": "max"},
        }
        result = parse_request(json.dumps(body).encode("utf-8"))
        assert result is not None
        assert result.output_config == {"effort": "max"}

    def test_output_config_defaults_to_none(self):
        """output_config defaults to None when not present."""
        body = {"model": "test"}
        result = parse_request(json.dumps(body).encode("utf-8"))
        assert result is not None
        assert result.output_config is None
