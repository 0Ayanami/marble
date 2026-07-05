import os
from pathlib import Path

import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message

from marble.llms.error_handler import api_calling_error_exponential_backoff


def load_provider_env(env_path: str = ".env") -> None:
    """Load active key-value settings from .env, ignoring commented lines."""
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().replace("-", "_").upper()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ[key] = value


def normalize_model_and_base_url(llm_model: str) -> tuple[str, Optional[str]]:
    base_url = os.environ.get("BASE_URL")
    env_model = os.environ.get("MODEL")
    model = llm_model or env_model or "gpt-3.5-turbo"
    has_together_key = bool(os.environ.get("TOGETHERAI_API_KEY"))
    if env_model and model.startswith(("gpt-", "openai/")):
        model = env_model
    if base_url and "/" not in model:
        model = f"openai/{model}"
    elif has_together_key and not model.startswith("together_ai/"):
        model = f"together_ai/{model}"
    return model, base_url


def resolve_api_key(base_url: Optional[str] = None) -> Optional[str]:
    if base_url:
        return os.environ.get("TOGETHERAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return None


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    """
    load_provider_env()
    llm_model, configured_base_url = normalize_model_and_base_url(llm_model)
    # litellm.set_verbose=True
    base_url = configured_base_url
    api_key = resolve_api_key(base_url)
    completion = litellm.completion(
        model=llm_model,
        messages=messages,
        max_tokens=max_token_num,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        base_url=base_url,
        api_key=api_key,
    )
    message_0: Message = completion.choices[0].message
    assert message_0 is not None
    assert isinstance(message_0, Message)
    return [message_0]
