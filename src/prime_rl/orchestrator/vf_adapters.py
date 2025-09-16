from __future__ import annotations

"""
Adapters to make Verifiers env parsing robust to non-vLLM backends (e.g., SGLang).

"""

from typing import Any, List, Tuple

from loguru import logger


def apply_verifiers_adapters(server_type: str) -> None:
    """Apply runtime patches to Verifiers for specific server types.
    """

    if server_type != "sglang":
        return

    try:
        # Delayed import to avoid hard dependency if verifiers isn't installed.
        from verifiers.envs.environment import Environment  # type: ignore
    except Exception as e:
        logger.debug(f"Verifiers not available for patching: {e}")
        return

    # Keep a reference to the original implementation in case we need it later
    _orig = getattr(Environment, "process_chat_format_vllm", None)
    if _orig is None:
        logger.debug("Environment.process_chat_format_vllm missing; skipping SGLang adapter")
        return

    async_warning_once = {"emitted": False}

    def _responses_have_logprobs(responses: list[Any]) -> bool:
        for response in responses:
            try:
                choices = getattr(response, "choices", None)
                if not choices:
                    continue
                choice = choices[0]
                logprobs = getattr(choice, "logprobs", None)
                content = getattr(logprobs, "content", None) if logprobs is not None else None
                if content:
                    return True
            except Exception:
                continue
        return False

    def _patched_process_chat_format_vllm(
        self: Any,
        prompt: list[dict[str, Any]],
        completion: list[dict[str, Any]],
        state: dict[str, Any],
        processing_class: Any,
        mask_env_responses: bool = False,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[float]]:
        """Patched version that preserves native logprobs when available."""
        responses = state["responses"]

        if _responses_have_logprobs(responses):
            return _orig(self, prompt, completion, state, processing_class, mask_env_responses)

        responses_idx = 0
        zipped = []
        for turn in completion:
            if turn["role"] == "assistant":
                zipped.append((turn, responses[responses_idx]))
                responses_idx += 1
            else:
                zipped.append((turn, None))
        assert len(responses) == responses_idx, "Responses not fully consumed"
        assert len(zipped) == len(completion), "Length mismatch"

        prompt_ids: list[int] = processing_class.apply_chat_template(
            conversation=prompt,  # type: ignore
            add_generation_prompt=True,
        )
        messages_consumed = [m for m in prompt]
        prompt_mask: list[int] = [0] * len(prompt_ids)
        completion_ids: list[int] = []
        completion_mask: list[int] = []
        completion_logprobs: list[float] = []

        i = 0
        while i < len(zipped):
            message, response = zipped[i]
            if message["role"] == "assistant":
                has_emitted_assistant = any(m.get("role") == "assistant" for m in messages_consumed)
                token_prefix: list[int] = processing_class.apply_chat_template(
                    conversation=messages_consumed,  # type: ignore
                    add_generation_prompt=(not has_emitted_assistant),
                )
                token_prefix_with_turn: list[int] = processing_class.apply_chat_template(
                    conversation=messages_consumed + [message],  # type: ignore
                )
                assert token_prefix_with_turn[: len(token_prefix)] == token_prefix, (
                    f"Token prefix mismatch. Token prefix: {token_prefix}, token prefix with turn: {token_prefix_with_turn}"
                )
                completion_turn_ids = token_prefix_with_turn[len(token_prefix) :]
                completion_turn_mask = [1] * len(completion_turn_ids)
                completion_turn_logprobs = [0.0] * len(completion_turn_ids)

                completion_ids.extend(completion_turn_ids)
                completion_mask.extend(completion_turn_mask)
                completion_logprobs.extend(completion_turn_logprobs)
                messages_consumed.append(message)
                i += 1

                if not async_warning_once["emitted"]:
                    logger.warning(
                        "SGLang response lacked logprobs; synthesizing zeroed values. Enable trainer.recompute_logprobs or upgrade the server."
                    )
                    async_warning_once["emitted"] = True
            else:
                assert message["role"] == "user" or message["role"] == "tool"
                consecutive_messages = [message]
                j = i + 1
                while j < len(zipped) and zipped[j][0]["role"] != "assistant":
                    consecutive_messages.append(zipped[j][0])
                    j += 1
                token_prefix: list[int] = processing_class.apply_chat_template(
                    conversation=messages_consumed  # type: ignore
                )
                token_prefix_with_turn: list[int] = processing_class.apply_chat_template(
                    conversation=messages_consumed + consecutive_messages,  # type: ignore
                )
                assert token_prefix_with_turn[: len(token_prefix)] == token_prefix, (
                    f"Token prefix mismatch. Token prefix: {token_prefix}, token prefix with turn: {token_prefix_with_turn}"
                )
                completion_turn_ids = token_prefix_with_turn[len(token_prefix) :]
                if mask_env_responses:
                    completion_turn_mask = [0] * len(completion_turn_ids)
                else:
                    completion_turn_mask = [1] * len(completion_turn_ids)
                completion_turn_logprobs = [0.0] * len(completion_turn_ids)
                completion_ids.extend(completion_turn_ids)
                completion_mask.extend(completion_turn_mask)
                completion_logprobs.extend(completion_turn_logprobs)
                messages_consumed.extend(consecutive_messages)
                i = j

        return (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        )

    # Apply the patch
    setattr(Environment, "process_chat_format_vllm", _patched_process_chat_format_vllm)
    logger.info("Applied Verifiers adapter: patched process_chat_format_vllm for SGLang backend")
