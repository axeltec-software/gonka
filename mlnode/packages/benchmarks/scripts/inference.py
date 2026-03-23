import sys
sys.path.append('../src')
sys.path.append('../../common/src')

import argparse
import os
import copy
import json
from time import sleep
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from validation.runner import run_validation, run_inference_only, run_validation_from_snapshots
from validation.prompts import preload_all_language_prompts, slice_mixed_language_prompts_with_langs
from validation.data import ModelInfo, RequestParams, ServerConfig, RunParams, InferenceValidationRun, load_snapshots_from_jsonl
from validation.model_presets import QWEN3_30B_INT4, QWEN3_30B_FP8
from validation.model_presets import GEMMA_3_27B_INT4, GEMMA_3_27B_FP8
from validation.model_presets import QWEN25_7B_AWQ, QWEN25_7B_INT8
from validation.model_presets import QWQ_32B_FP8, QWQ_32B_INT4

N_PROMPTS = 1000
MAX_WORKERS = None
run_params_high_temp = RunParams(
    # exp_name='gemma27B',
    exp_name='qwq_32b',
    # exp_name='qwen3_30b',
    output_path='data/inference_results',
    n_prompts=N_PROMPTS,
    timeout=1800,
    tokenizer_model_name='unsloth/llama-3-8b-Instruct',
    request=RequestParams(
        max_tokens=3000,
        temperature=0.99,
        seed=42,
        top_logprobs=5,
    ),
)

def get_run_params(temp, prompts):
    params = copy.deepcopy(run_params_high_temp)
    params.request.temperature = temp
    params.n_prompts = prompts
    return params


server_1xH100_1 = ServerConfig(
    ip='86.38.238.14',
    inference_port='8001',
    node_port='17340',
    gpu='1xH100',
)

server_1xH100_2 = ServerConfig(
    ip='86.38.238.14',
    inference_port='8000',
    node_port='19145',
    gpu='1xH100',
)
# curl -X GET "http://151.237.25.234:22657/health"
# 151.237.25.234:22657
server_4x3090_1 = ServerConfig(
    ip='151.237.25.234',
    inference_port='22657',
    node_port='24434',
    gpu='4x3090',
)

server_4x3090_2 = ServerConfig(
    ip='109.248.7.144',
    inference_port='20347',
    node_port='20505',
    gpu='4x3090',
)

# honest_preset = QWEN25_7B_INT8
# fraudulent_preset = QWEN25_7B_AWQ
honest_preset = QWQ_32B_FP8
fraudulent_preset = QWQ_32B_INT4
# fraudulent_preset = QWEN3_30B_INT4
# honest_preset = QWEN3_30B_FP8

langs = ("en", "sp","ch", "hi", "ar")
runs = [
    # Honest FP8 on 1xH100 vs FP8 on 1xH100
    InferenceValidationRun(
        model_inference=honest_preset,
        model_validation=honest_preset,
        server_inference=server_1xH100_1,
        server_validation=server_1xH100_2,
        # run_inference=get_run_params(0.99, N_PROMPTS),
        # run_validation=get_run_params(0.99, N_PROMPTS),
        run_inference=get_run_params(0.99, N_PROMPTS//5),
        run_validation=get_run_params(0.99, N_PROMPTS//5),
        max_workers=MAX_WORKERS,
    ),
    # Fradulent INT4 on 1xH100 vs FP8 on 1xH100
    # InferenceValidationRun(
    #     model_inference=fraudulent_preset,
    #     model_validation=honest_preset,
    #     server_inference=server_1xH100_2,
    #     server_validation=server_1xH100_1,
    #     run_inference=get_run_params(0.7, N_PROMPTS//5),
    #     run_validation=get_run_params(0.7, N_PROMPTS//5),
    #     max_workers=MAX_WORKERS,
    # ),
    # # Fradulent INT4 on 4x3090 vs FP8 on 1xH100
    # InferenceValidationRun(
    #     model_inference=fraudulent_preset,
    #     model_validation=honest_preset,
    #     server_inference=server_4x3090_2,
    #     server_validation=server_1xH100_1,
    #     run_inference=get_run_params(0.7, N_PROMPTS//5),
    #     run_validation=get_run_params(0.7, N_PROMPTS//5),
    #     max_workers=MAX_WORKERS,
    # ),
    # # Fradulent INT4 on 4x3090 vs FP8 on 4x3090
    # InferenceValidationRun(
    #     model_inference=fraudulent_preset,
    #     model_validation=honest_preset,
    #     server_inference=server_4x3090_2,
    #     server_validation=server_4x3090_1,
    #     run_inference=get_run_params(0.7, N_PROMPTS//5),
    #     run_validation=get_run_params(0.7, N_PROMPTS//5),
    #     max_workers=MAX_WORKERS,
    # ),
    # # Honest FP8 on 4x3090 vs FP8 on 1xH100
    # InferenceValidationRun(
    #     model_inference=honest_preset,
    #     model_validation=honest_preset,
    #     server_inference=server_4x3090_1,
    #     server_validation=server_1xH100_1,
    #     run_inference=get_run_params(0.99, N_PROMPTS//5),
    #     run_validation=get_run_params(0.99, N_PROMPTS//5),
    #     max_workers=MAX_WORKERS,
    # ),
    # # Honest FP8 on 4x3090 vs FP8 on 4x3090
    # InferenceValidationRun(
    #     model_inference=honest_preset,
    #     model_validation=honest_preset,
    #     server_inference=server_4x3090_1,
    #     server_validation=server_4x3090_2,
    #     run_inference=get_run_params(0.99, N_PROMPTS//5),
    #     run_validation=get_run_params(0.99, N_PROMPTS//5),
    #     max_workers=MAX_WORKERS,
    # ),
]


def post_up(server: ServerConfig, payload, timeout: int):
    response = requests.post(server.up_url(), json=payload, timeout=timeout)
    response.raise_for_status()
    return server, response


def _snapshot_path(data_path: str) -> str:
    """Derive the inference-snapshot file path from the results file path."""
    base = data_path[:-5] if data_path.endswith('.jsonl') else data_path
    return f"{base}_snapshots.jsonl"


def _build_model_infos(cfg: InferenceValidationRun):
    inference_model_info = ModelInfo(
        url=cfg.server_inference.inference_url(),
        name=cfg.model_inference.model,
        deploy_params={
            "GPU": cfg.server_inference.gpu,
            "precision": cfg.model_inference.precision,
        },
    )
    validation_model_info = ModelInfo(
        url=cfg.server_validation.inference_url(),
        name=cfg.model_validation.model,
        deploy_params={
            "GPU": cfg.server_validation.gpu,
            "precision": cfg.model_validation.precision,
        },
    )
    return inference_model_info, validation_model_info


def _save_config(cfg: InferenceValidationRun, output_dir: str, setting_name: str) -> None:
    config_filename = f"{setting_name.rsplit('.', 1)[0]}_config.json"
    config_path = f"{output_dir}/{config_filename}"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference/validation benchmark.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'inference', 'validate'],
        default='full',
        help=(
            "full       – run inference + validation together (default)\n"
            "inference  – run inference only; save snapshots for later validation\n"
            "validate   – run validation only from a previously saved snapshots file\n"
            "             (requires --snapshots)"
        ),
    )
    parser.add_argument(
        '--snapshots',
        type=str,
        default=None,
        metavar='FILE',
        help="Path to a snapshots .jsonl file produced by --mode inference. Required for --mode validate.",
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        metavar='FILE',
        help="Override output path for results (only used with --mode validate).",
    )
    args = parser.parse_args()

    if args.mode == 'validate':
        if not args.snapshots:
            parser.error("--snapshots FILE is required when --mode validate is used.")
        snapshots = load_snapshots_from_jsonl(args.snapshots)
        if not snapshots:
            print("No snapshots found in file, nothing to do.")
            return
        if args.output:
            output_path = args.output
        else:
            base = args.snapshots
            if base.endswith('_snapshots.jsonl'):
                output_path = base[: -len('_snapshots.jsonl')] + '_validated.jsonl'
            else:
                output_path = base.replace('.jsonl', '_validated.jsonl')
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        max_workers = snapshots[0].request_params.retries_max_attempts  # reuse workers from first snap or default
        _ = run_validation_from_snapshots(
            snapshots,
            max_workers=10,
            output_path=output_path,
        )
        print(f"Validation complete. Results saved to: {output_path}")
        return

    dataset = preload_all_language_prompts(langs=langs)

    for cfg in runs:
        inference_model_info, validation_model_info = _build_model_infos(cfg)
        os.makedirs(cfg.run_inference.output_path, exist_ok=True)
        request_params = cfg.run_inference.request
        setting_name = cfg.setting_filename()
        data_path = f"{cfg.run_inference.output_path}/{setting_name}"
        n_prompts = cfg.run_inference.n_prompts

        _save_config(cfg, cfg.run_inference.output_path, setting_name)

        prompts, languages = slice_mixed_language_prompts_with_langs(
            dataset, per_language_n=n_prompts // len(langs), langs=langs
        )

        if args.mode == 'full':
            _ = run_validation(
                prompts,
                languages=languages,
                inference_model=inference_model_info,
                validation_model=validation_model_info,
                request_params=request_params,
                max_workers=cfg.max_workers,
                output_path=data_path,
            )
            print(f"Completed run. Results saved to: {data_path}")

        elif args.mode == 'inference':
            snapshot_path = _snapshot_path(data_path)
            _ = run_inference_only(
                prompts,
                inference_model=inference_model_info,
                validation_model=validation_model_info,
                request_params=request_params,
                max_workers=cfg.max_workers,
                output_path=snapshot_path,
                languages=languages,
            )
            print(f"Inference complete. Snapshots saved to: {snapshot_path}")
            print(f"To run validation later:  python inference.py --mode validate --snapshots {snapshot_path}")


if __name__ == "__main__":
    main()