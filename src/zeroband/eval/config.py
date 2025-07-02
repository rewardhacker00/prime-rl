from pathlib import Path
from typing import Annotated

from pydantic import Field

from zeroband.training.orchestrator.config import (
    ClientConfig,
    LogConfig,
    SamplingConfig,
)
from zeroband.training.orchestrator.config import (
    EvalConfig as OrchestratorEvalConfig,
)
from zeroband.utils.config import ModelConfig, MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseSettings


class EvalConfig(BaseSettings):
    """Configures evaluation."""

    # The client configuration
    client: Annotated[ClientConfig, Field(default=ClientConfig())]

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The sampling configuration
    sampling: Annotated[SamplingConfig, Field(default=SamplingConfig())]

    # The evaluation configuration
    eval: Annotated[OrchestratorEvalConfig, Field(default=OrchestratorEvalConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    # The logging configuration
    log: Annotated[LogConfig, Field(default=LogConfig(path=Path("logs/eval")))]

    use_tqdm: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to use tqdm to display progress bars during model generation.",
        ),
    ]
