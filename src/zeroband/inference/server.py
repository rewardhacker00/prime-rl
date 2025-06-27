import uvloop
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser,
    make_arg_parser,
    run_server,
    validate_parsed_serve_args,
)

from zeroband.inference.config import InferenceConfig
from zeroband.utils.pydantic_config import parse_argv


def server(config: InferenceConfig, vllm_args: list[str]):
    # Translate configs
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=vllm_args, namespace=config.to_vllm())
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)
    server(config, vllm_args=config.get_unknown_args())


if __name__ == "__main__":
    main()
