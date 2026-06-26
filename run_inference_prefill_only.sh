python ./mlnode/packages/benchmarks/scripts/inference_validation/inference_hacked.py \
    --url http://0.0.0.0:8005 \
    --model Qwen/Qwen3-0.6B \
    --n-prompts 100 \
    --max-tokens 1 \
    --top-logprobs 4 \
    --prompts-file /home/irene/projects/gonka_2stage_val/gonka/mlnode/packages/benchmarks/data/experiments/inference_fraud_Qwen600M-fp8_hack/prompts.txt \
    --config-file /home/irene/projects/gonka_2stage_val/gonka/mlnode/packages/benchmarks/data/experiments/inference_fraud_Qwen600M-fp8_hack/inference_config.json \
    #--no-token-str \
