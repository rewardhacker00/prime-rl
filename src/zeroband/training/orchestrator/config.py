from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from zeroband.training.config import PathConfig
from zeroband.utils.config import MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class OAIClientConfig(BaseConfig):
    """Configures the client to be used for inference."""

    base_url: Annotated[str, Field(default="http://localhost:8000/v1", description="Base URL of the OpenAI API.")]

    api_key: Annotated[str, Field(default="insecure", description="API key to use for the OpenAI API.")]


class CompletionConfig(BaseConfig):
    """Configures the completion request sent to the inference pool."""

    model: Annotated[str, Field(default="Qwen/Qwen3-0.6B", description="Name or path of the model to use.")]

    temperature: Annotated[
        float,
        Field(
            default=1.0,
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily.",
        ),
    ]

    max_tokens: Annotated[
        int | None,
        Field(default=None, description="Maximum number of tokens to generate. If None, will use the model's default max length."),
    ]


# TODO(Mika, Will): Change data config to verifiers environment
# TODO(Mika): Find an elegant way to enable online/ offline difficulty filtering
class DataConfig(BaseConfig):
    """Configures the data to be used for inference."""

    name: Annotated[
        str,
        Field(
            default="PrimeIntellect/INTELLECT-2-RL-Dataset",
            description="Name of the HF dataset to use.",
        ),
    ]

    split: Annotated[str, Field(default="train", description="Split of the dataset to use.")]


class LogConfig(BaseConfig):
    """Configures the logger."""

    level: Annotated[
        Literal["debug", "info"],
        Field(default="info", description="Logging level for the inference run. Will determine the logging verbosity and format."),
    ]

    all_ranks: Annotated[
        bool, Field(default=False, description="Whether to log from all DP ranks. If False, will only log from the main rank (DP rank 0).")
    ]

    utc: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to use UTC time in the logger. If False, it will default to the local time. If the local time is wrong, you can set it by setting the `TZ` environment variable. For example, `TZ=America/Los_Angeles` will set the local time to SF time.",
        ),
    ]


class OrchestratorConfig(BaseSettings):
    """Configures the orchestrator for RL training."""

    # The OAI client configuration
    client: Annotated[OAIClientConfig, Field(default=OAIClientConfig())]

    # The completion configuration
    completion: Annotated[CompletionConfig, Field(default=CompletionConfig())]

    # The data configuration
    data: Annotated[DataConfig, Field(default=DataConfig())]

    # The logging configuration
    log: Annotated[LogConfig, Field(default=LogConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    # The training orchestration configuration
    batch_size: Annotated[int, Field(default=128, description="Number of samples to process in each batch.")]

    samples_per_problem: Annotated[int, Field(default=1, description="Number of samples to process for each problem.")]

    max_steps: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum number of training steps to run. If None, will run indefinitely.",
        ),
    ]

    async_level: Annotated[
        int,
        Field(
            default=2,
            description="Maximum number of async levels to use. If 0, will do synchronous RL. Else, it will allow to go `async_level` steps ahead of training.",
        ),
    ]

    rollout: Annotated[
        PathConfig,
        Field(
            default=PathConfig(path=Path("rollouts")),
            description="Path to write inference outputs to. Will be populated by the orchestrator with responses from inference pool.",
        ),
    ]

    checkpoints: Annotated[
        PathConfig,
        Field(
            default=PathConfig(path=Path("checkpoints")),
            description="Path to read new model checkpoints from. Will be populated by the trainer.",
        ),
    ]

    seed: Annotated[int | None, Field(default=None, description="Random seed for the orchestrator.")]

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.samples_per_problem != 0:
            raise ValueError("Batch size must be divisible by the number of samples per problem")
        return self
