from safetensors.torch import safe_open


class WeightUpdaterWorker:
    """
    This worker is used to update the weights of the model.

    Its meant to be used as a worker extension, so it has access to the model runner.

    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        worker_extension_cls="zeroband.inference.vllm_worker.WeightUpdaterWorker",
    )

    """

    def update_weight(self, ckpt_path):
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            # Create a better weight iterator that filters out empty keys and handles prefixes
            def weights_iterator():
                for key in f.keys():
                    # Skip empty keys
                    if not key:
                        continue
                    yield key, f.get_tensor(key)

            # Load weights
            self.model_runner.model.load_weights(weights_iterator())

        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        # loading here to avoid import at the top of the file

        # Process weights after loading (important for some models)
        device = next(self.model_runner.model.parameters()).device
        process_weights_after_loading(
            self.model_runner.model, self.model_runner.model_config, device
        )
        print("weight updated")
