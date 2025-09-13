from prime_rl.inference.backends.vllm import VLLMBackend
from prime_rl.inference.config import InferenceConfig
from prime_rl.utils.pydantic_config import parse_argv

BACKENDS = {"vllm": VLLMBackend}


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)
    server_type = config.server.server_type
    backend_cls = BACKENDS.get(server_type)
    if backend_cls is None:
        raise ValueError(f"Unsupported server type: {server_type}")
    backend = backend_cls()
    backend.startup(config)


if __name__ == "__main__":
    main()
