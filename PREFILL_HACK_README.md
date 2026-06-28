## Prefill attack reproducing (on an example of Qwen/Qwen3-0.6B vs Qwen/Qwen3-0.6B-FP8)
- Run a server with a small model:
```
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B-FP8 \
    --max-model-len 2048 \
    --port 8005 \
    --gpu-memory-utilization 0.7 \
    --no-async-scheduling \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --poc-decode \
    --poc-share 0.5 \
    --poc-max-batch-size 32 \
    --poc-seq-len 256 \
    --poc-max-tokens 256 \
```
- When the server is up, execute the script `./run_inference_fraud.sh`.
  This will result in the desired output decoded token sequence 
- Rename the output directory with the obtained result to  `./mlnode/packages/benchmarks/data/experiments/inference_fraud_fp8_hack`
- Stop the running server and run a new server with a large model:
```
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --max-model-len 2048 \
    --port 8005 \
    --gpu-memory-utilization 0.7 \
    --no-async-scheduling \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --poc-decode \
    --poc-share 0.5 \
    --poc-max-batch-size 32 \
    --poc-seq-len 256 \
    --poc-max-tokens 256 \
```
- When the server is up, execute the script `./run_inference_prefill_only.sh`.
  This will generate new prompts (from the old prompts + decoded texts), put them to a file `./mlnode/packages/benchmarks/data/experiments/inference_fraud_fp8_hack/prompts.txt`, and execute a prefill run (with a single decoded token) on the large model with the new prompts. The inference results will be put to a new folder in  `./mlnode/packages/benchmarks/data/experiments/`. It should be renamed to `./mlnode/packages/benchmarks/data/experiments/inference_fraud_hack`
- Do a standard validation run with the large model using the script `./run_validation.sh`. The validation results will be put to `./mlnode/packages/benchmarks/data/experiments/inference_fraud_hack`
- Do a standard inference-validation run sequence for the fraud scenario to compare with. The results should be put to a folder `./mlnode/packages/benchmarks/data/experiments/inference_fraud`. 
- Run a jupyter.notebook `mlnode/packages/benchmarks/notebooks/analysis_new.ipynb` to see the graphycal results.
