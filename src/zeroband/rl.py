from pprint import pprint

from zeroband.utils.pydantic_config import BaseConfig, BaseSettings, parse_argv


class TrainConfig(BaseConfig):
    name: str


class EvalConfig(BaseConfig):
    name: str


class RLConfig(BaseSettings):
    train: TrainConfig
    eval: EvalConfig


if __name__ == "__main__":
    config = parse_argv(RLConfig)
    pprint(config)
