from __future__ import annotations

import os
import uuid
from typing import Any, Optional

from langchain_openai import ChatOpenAI


class ChatOpenAICompatible(ChatOpenAI):
    request_id_header: str = "X-Request-ID"
    request_id: Optional[str] = None

    def __init__(
        self,
        model: str = "gpt-oss:120b",
        request_id: Optional[str] = None,
        request_id_header: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        # Prepare values FIRST (local variables only)
        rid = request_id or str(uuid.uuid4())
        header_name = (
            request_id_header
            or os.environ.get("CUSTOM_LLM_REQUEST_ID_HEADER", "X-Request-ID")
        )

        default_headers = {header_name: rid}

        # Call parent FIRST
        super().__init__(
            model=model,
            base_url=base_url or os.environ.get("CUSTOM_LLM_BASE_URL"),
            api_key=api_key or os.environ.get("CUSTOM_LLM_KEY"),
            default_headers=default_headers,
            **kwargs,
        )

        # THEN assign attributes
        self.request_id = rid
        self.request_id_header = header_name

    @property
    def _llm_type(self) -> str:
        return "openai_compatible"