import asyncio

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pyarrow import Table

from zeroband.training.orchestrator.config import CompletionConfig
from zeroband.training.parquet import SCHEMA
from zeroband.utils.logger import get_logger


async def health_check(client: AsyncOpenAI, timeout: int = 60, interval: int = 10) -> None:
    logger = get_logger()
    logger.info("Checking health of inference pool")
    num_attempts = 0
    while num_attempts * interval < timeout:
        try:
            await client.models.list()
            logger.success("Inference pool is ready")
            return
        except Exception as e:
            num_attempts += 1
            logger.warning(f"Inference pool cannot be reached after {num_attempts} attempt(s) (Error: {e})")
            await asyncio.sleep(interval)
    msg = f"Inference pool is not ready after {num_attempts} attempt(s). Aborting..."
    logger.error(msg)
    raise TimeoutError(msg)


async def generate_completion(
    client: AsyncOpenAI, completion_config: CompletionConfig, ckpt_step: int, messages: list[dict[str, str]]
) -> ChatCompletion:
    config = completion_config.model_copy()
    config.model = f"{config.model}-{ckpt_step}" if ckpt_step > 0 else config.model
    response = await client.chat.completions.create(
        **config.model_dump(),
        messages=messages,
    )
    assert len(response.choices) == 1, "Response should always have one choice"
    return response


# "rewards",
# "task_rewards",
# "length_penalties",
# "target_lengths",
# "task_type",


def get_parquet(completions: list[ChatCompletion], rewards: list[float]) -> Table:
    rows = []
    for completion, reward in zip(completions, rewards):
        rows.append(
            {
                "prompt": completion.choices[0].message.content,
                "completion": completion.choices[0].message.content,
                "reward": reward,
            }
        )
    return Table.from_pylist(rows, schema=SCHEMA)
