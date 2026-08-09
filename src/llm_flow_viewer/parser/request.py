"""Request body parser for LLM API requests.

Parses JSON request bodies into ParsedRequest dataclasses,
extracting model, max_tokens, messages, tools, system prompts,
thinking config, and other fields.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from llm_flow_viewer.parser.models import ParsedRequest

logger = logging.getLogger(__name__)


def parse_request(
    raw: bytes | None,
    *,
    include_messages: bool = True,
    include_tools: bool = True,
    include_system: bool = True,
) -> ParsedRequest | None:
    """Parse a byte-encoded JSON request body into a ParsedRequest.

    Args:
        raw: The raw byte payload from the HTTP request.
        include_messages: If True, populate the messages field.
        include_tools: If True, populate the tools field.
        include_system: If True, populate the system field.

    Returns:
        A ParsedRequest instance, or None if the content is empty or
        the JSON cannot be decoded.
    """
    if not raw:
        return None

    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Failed to decode request body as JSON: %s", e)
        return None

    return ParsedRequest(
        model=data.get("model", ""),
        max_tokens=data.get("max_tokens", 0),
        messages=data.get("messages", []) if include_messages else [],
        tools=data.get("tools", []) if include_tools else [],
        system=data.get("system", []) if include_system else [],
        output_config=data.get("output_config"),
        thinking=data.get("thinking"),
        stream=data.get("stream", False),
    )
