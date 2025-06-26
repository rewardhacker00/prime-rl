import pyarrow as pa

# Parquet schema used to communicate between orchestrator and trainer
SCHEMA = pa.schema(
    [
        ("input_tokens", pa.list_(pa.int32())),
        ("output_tokens", pa.list_(pa.int32())),
        ("input_logprobs", pa.list_(pa.float32())),  # Optional - can be null
        ("output_logprobs", pa.list_(pa.float32())),  # Optional - can be null
        ("advantages", pa.float32()),
        ("rewards", pa.float32()),
        ("task_rewards", pa.float32()),
        ("length_penalties", pa.float32()),
        ("target_lengths", pa.int32()),
        ("task_type", pa.string()),
        ("temperature", pa.float32()),
    ]
)
