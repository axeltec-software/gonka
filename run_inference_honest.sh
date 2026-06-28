python ./mlnode/packages/benchmarks/scripts/inference_validation/inference.py \
    --url http://0.0.0.0:8005 \
    --model Qwen/Qwen3-0.6B \
    --n-prompts 100 \
    --max-tokens 300 \
    --top-logprobs 4 
