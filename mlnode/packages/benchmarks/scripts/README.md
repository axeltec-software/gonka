# Inference & Validation Script

`inference.py` runs LLM inference and prefill validation against vLLM servers. It supports parallel (simultaneous) and sequential (step-by-step) workflows.

## Prerequisites

Run from the `scripts/` directory. The script expects vLLM inference and validation servers to be running at the addresses configured in the `ServerConfig` definitions inside the script; other serving parameters can be updated in the structure `RunParams`. Configuration settings for the models being served should be updated correspondingly inside the script `../src/validation/model_presets.py`. 

## Usage

```bash
python3 inference.py --enforced-tokens-in <vllm|payload> --type <parallel|sequential> [options]
```

### Arguments

| Argument | Values | Default | Description |
|---|---|---|---|
| `--enforced-tokens-in` | `vllm`, `payload` | `payload` | `vllm` — customized vLLM with prefill validation support; `payload` — original vLLM (enforced tokens appended in the request payload) |
| `--type` | `parallel`, `sequential` | `parallel` | `parallel` — inference and validation run simultaneously; `sequential` — run one after the other |
| `--mode` | `none`, `inference`, `validation`, `both` | `none` | Required for `--type=sequential`. Which step(s) to execute |
| `--inference-responses` | path | — | Path to a `.jsonl` file with saved inference responses. Required when `--type=sequential` and `--mode=validation` |
| `--connection-error-wait` | int (seconds) | `120` | Wait time before retrying after a connection error |
| `--max-retries` | int | `3` | Max retry attempts on connection errors |

### Examples

**Parallel workflow with vLLM vanilla version** — run inference and validation together (both servers must be up):

```bash
python3 inference.py --type parallel --enforced-tokens-in payload
```

**Sequential workflow with vLLM vanilla version** — inference only:

```bash
python3 inference.py --type sequential --mode inference --enforced-tokens-in payload
```

**Sequential workflow with vLLM vanilla version** — validation from saved inference responses:

```bash
python3 inference.py --type sequential --mode validation \
    --enforced-tokens-in payload \
    --inference-responses data/inference_results/<name>_inference_responses.jsonl
```

**Sequential workflow with customized vLLM allowing for prefill validation support** — both steps back-to-back:

```bash
python3 inference.py --type sequential --mode both --enforced-tokens-in vllm
```

## Output

Results are saved to `data/inference_results/`. Each run produces:
- `*_config.json` — run configuration snapshot
- `*_inference_responses.jsonl` — intermediate inference results (sequential mode)
- `*.jsonl` — final validation results

## Configuration

Model presets, server addresses, run parameters (temperature, number of prompts, languages), and run definitions are configured directly in the script at the top of the file. Configuration settings for the models being served can be updated inside the script `../src/validation/model_presets.py`.
