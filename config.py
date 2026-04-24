import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

class Configuration(BaseModel):
    @staticmethod
    def _ollama_override():
        """Return True if Ollama should be used for all models."""
        return os.environ.get("USE_OTHER_LLM", "false").lower() == "true"

    @staticmethod
    def _ollama_model():
        """Return the Ollama model name to use (default: 'gpt-oss:20b')."""
        return os.environ.get("OTHER_LLM_MODEL", "gpt-oss:20b")

    @staticmethod
    def _ollama_base_url():
        """Return the Ollama base URL to use (default: 'https://ollama.com')."""
        return os.environ.get("OTHER_LLM_BASE_URL", "https://ollama.com")

    @staticmethod
    def _ollama_api_key():
        """Return the Ollama API key if set, otherwise 'ollama' as placeholder."""
        return os.environ.get("OLLAMA_API_KEY", "ollama")

    @staticmethod
    def _bedrock_override():
        """Return True if AWS Bedrock should be used for all models."""
        return os.environ.get("USE_BEDROCK", "false").lower() == "true"

    @staticmethod
    def _bedrock_model():
        """Return the Bedrock model ID to use (default: 'gpt-oss:20b')."""
        return os.environ.get("BEDROCK_MODEL", "gpt-oss:20b")
    
    @staticmethod
    def _bedrock_region():
        """Return the AWS region for Bedrock (default: 'us-east-1')."""
        return os.environ.get("BEDROCK_REGION", "us-east-1")
    
    @staticmethod
    def _enforce_knn_tool():
        """Return True if KNN tool should be enforced for unmatched tool calls."""
        return os.environ.get("ENFORCE_KNN_TOOL", "false").lower() == "true"

    """The configuration for the agent."""

    query_generator_provider: str = Field(
        default="openai",
        metadata={
            "description": "The name of the language model provider to use for the agent's query generation."
        },
    )

    query_generator_model: str = Field(
        default="gpt-4.1-mini-2025-04-14",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_provider: str = Field(
        default="openai",
        metadata={
            "description": "The name of the language model provider to use for the agent's reflection."
        },
    )

    reflection_model: str = Field(
        default="gpt-4.1-mini-2025-04-14",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_provider: str = Field(
        default="openai",
        metadata={
            "description": "The name of the language model provider to use for the agent's answer."
        },
    )

    answer_model: str = Field(
        default="gpt-4.1-mini-2025-04-14",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    reasoning_provider: str = Field(
        default="openai",
        metadata={
            "description": "The name of the model provider to use for the agent's reasoning steps."
        },
    )

    reasoning_model: str = Field(
        default="gpt-4.1-mini-2025-04-14",
        metadata={
            "description": "The name of the model to use for the agent's reasoning steps."
        },
    )

    number_of_initial_queries: int = Field(
        default=2,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=1,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    max_replans: int = Field(
        default=5,
        metadata={"description": "The number of replanning steps for the planning graph."},
    )

    planner_provider: str = Field(
        default="openai",
        metadata={
            "description": "The name of the language model provider to use for the agent's answer."
        },
    )

    planner_model: str = Field(
        default="gpt-4.1-mini-2025-04-14",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    replanner_provider: str = Field(
        default="openai",
        metadata={
            "description": "The name of the language model provider to use for the agent's answer."
        },
    )

    replanner_model: str = Field(
        default="gpt-4.1-mini-2025-04-14",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    tool_provider: str = Field(
        default="openai",
        metadata={
            "description": "The name of the language model provider to use for the agent's answer."
        },
    )

    tool_model: str = Field(
        default="gpt-4.1-mini-2025-04-14",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )


    def get_answer_model(self, **kwargs):
        if self._ollama_override():
            return ChatOpenAI(
                model=self._ollama_model(),
                openai_api_key=self._ollama_api_key(),
                openai_api_base=self._ollama_base_url(),
                **kwargs
            )
        if self._bedrock_override():
            return init_chat_model(model=self._bedrock_model(), model_provider="bedrock", region_name=self._bedrock_region(), **kwargs)
        return init_chat_model(
            model=self.answer_model,
            model_provider=self.answer_provider,
            **kwargs
        )

    def get_query_generator_model(self, **kwargs):
        if self._ollama_override():
            return ChatOpenAI(
                model=self._ollama_model(),
                openai_api_key=self._ollama_api_key(),
                openai_api_base=self._ollama_base_url(),
                **kwargs
            )
        if self._bedrock_override():
            return init_chat_model(model=self._bedrock_model(), model_provider="bedrock", region_name=self._bedrock_region(), **kwargs)
        return init_chat_model(
            model=self.query_generator_model,
            model_provider=self.query_generator_provider,
            **kwargs
        )

    def get_reasoning_model(self, **kwargs):
        if self._ollama_override():
            return ChatOpenAI(
                model=self._ollama_model(),
                openai_api_key=self._ollama_api_key(),
                openai_api_base=self._ollama_base_url(),
                **kwargs
            )
        if self._bedrock_override():
            return init_chat_model(model=self._bedrock_model(), model_provider="bedrock", region_name=self._bedrock_region(), **kwargs)
        return init_chat_model(
            model=self.reasoning_model,
            model_provider=self.reasoning_provider,
            **kwargs
        )

    def get_reflection_model(self, **kwargs):
        if self._ollama_override():
            return ChatOpenAI(
                model=self._ollama_model(),
                openai_api_key=self._ollama_api_key(),
                openai_api_base=self._ollama_base_url(),
                **kwargs
            )
        if self._bedrock_override():
            return init_chat_model(model=self._bedrock_model(), model_provider="bedrock", region_name=self._bedrock_region(), **kwargs)
        return init_chat_model(
            model=self.reflection_model,
            model_provider=self.reflection_provider,
            **kwargs
        )

    def get_planner_model(self, **kwargs):
        if self._ollama_override():
            return ChatOpenAI(
                model=self._ollama_model(),
                openai_api_key=self._ollama_api_key(),
                openai_api_base=self._ollama_base_url(),
                **kwargs
            )
        if self._bedrock_override():
            return init_chat_model(model=self._bedrock_model(), model_provider="bedrock", region_name=self._bedrock_region(), **kwargs)
        return init_chat_model(
            model=self.planner_model,
            model_provider=self.planner_provider,
            **kwargs
        )

    def get_replanner_model(self, **kwargs):
        if self._ollama_override():
            return ChatOpenAI(
                model=self._ollama_model(),
                openai_api_key=self._ollama_api_key(),
                openai_api_base=self._ollama_base_url(),
                **kwargs
            )
        if self._bedrock_override():
            return init_chat_model(model=self._bedrock_model(), model_provider="bedrock", region_name=self._bedrock_region(), **kwargs)
        return init_chat_model(
            model=self.replanner_model,
            model_provider=self.replanner_provider,
            **kwargs
        )

    def get_tool_model(self, **kwargs):
        if self._ollama_override():
            return ChatOpenAI(
                model=self._ollama_model(),
                openai_api_key=self._ollama_api_key(),
                openai_api_base=self._ollama_base_url(),
                **kwargs
            )
        if self._bedrock_override():
            return init_chat_model(model=self._bedrock_model(), model_provider="bedrock", region_name=self._bedrock_region(), **kwargs)
        return init_chat_model(
            model=self.tool_model,
            model_provider=self.tool_provider,
            **kwargs
        )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
