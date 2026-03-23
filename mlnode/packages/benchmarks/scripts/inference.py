import sys
sys.path.append('../src')
sys.path.append('../../common/src')

import os
import copy
import json
from time import sleep
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

from validation.runner import run_validation
from validation.prompts import preload_all_language_prompts, slice_mixed_language_prompts_with_langs
from validation.data import ModelInfo, RequestParams, ServerConfig, RunParams, InferenceValidationRun
from validation.model_presets import QWEN3_30B_INT4, QWEN3_30B_FP8
from validation.model_presets import GEMMA_3_27B_INT4, GEMMA_3_27B_FP8
from validation.model_presets import QWEN25_7B_AWQ, QWEN25_7B_INT8, QWEN3_600M_FP8, QWEN3_600M_FP16

N_PROMPTS = 1000
MAX_WORKERS = None
run_params_high_temp = RunParams(
    exp_name='qwen3-600m',
    output_path='data/inference_results',
    n_prompts=N_PROMPTS,
    timeout=1800,
    #tokenizer_model_name='Qwen/Qwen2-1.5B-Instruct',
    request=RequestParams(
        max_tokens=3000,
        temperature=0.99,
        seed=42,
        top_logprobs=4,
    ),
)

def get_run_params(temp, prompts):
    params = copy.deepcopy(run_params_high_temp)
    params.request.temperature = temp
    params.n_prompts = prompts
    return params


server_1xH100_1 = ServerConfig(
    ip='80.188.223.202',
    inference_port='17390',
    node_port='17340',
    gpu='1xH100',
)

server_1xH100_2 = ServerConfig(
    ip='80.188.223.202',
    inference_port='19477',
    node_port='19145',
    gpu='1xH100',
)

server_4x3090_1 = ServerConfig(
    ip='163.5.212.39',
    inference_port='18906',
    node_port='24434',
    gpu='4x3090',
)

server_4x3090_2 = ServerConfig(
    ip='109.248.7.144',
    inference_port='20347',
    node_port='20505',
    gpu='4x3090',
)

server_0_3xRTX4000_1 = ServerConfig(
    ip='0.0.0.0',
    inference_port='8005',
    node_port='17340',
    gpu='0.3xRTX4000',
)

server_0_3xRTX4000_2 = ServerConfig(
    ip='0.0.0.0',
    inference_port='8010',
    node_port='19145',
    gpu='0.3xRTX4000',
)

honest_preset = QWEN3_600M_FP16
fraudulent_preset = QWEN3_600M_FP8

langs = ("en", "sp","ch", "hi", "ar")
#langs = ("en",)
runs = [
    # Honest FP8 on 1xH100 vs FP8 on 1xH100
    InferenceValidationRun(
        model_inference=fraudulent_preset,
        model_validation=honest_preset,
        server_inference=server_0_3xRTX4000_1,
        server_validation=server_0_3xRTX4000_2,
        run_inference=get_run_params(0.99, N_PROMPTS),
        run_validation=get_run_params(0.99, N_PROMPTS),
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


def main():
    dataset = preload_all_language_prompts(langs=langs)
    for cfg in runs:
        # tokenizer_model_name = cfg.run_inference.tokenizer_model_name
        # _ = AutoTokenizer.from_pretrained(tokenizer_model_name)

        # os.makedirs(cfg.run_inference.output_path, exist_ok=True)

        # inference_payload = cfg.model_inference.to_deploy_payload()
        # validation_payload = cfg.model_validation.to_deploy_payload()
        # up_requests = [
        #     (cfg.server_inference, inference_payload),
        #     (cfg.server_validation, validation_payload),
        # ]
        # print(up_requests)
        # with ThreadPoolExecutor(max_workers=len(up_requests)) as executor:
        #     futures = [
        #         executor.submit(post_up, srv, payload, cfg.run_inference.timeout)
        #         for srv, payload in up_requests
        #     ]
        #     for future in futures:
        #         _, response = future.result()
        #         print(response.status_code)
        #         print(response.text)

        # sleep(5)

        inference_model_info = ModelInfo(
            url=cfg.server_inference.inference_url(),
            #url="http://0.0.0.0:8005",
            name=cfg.model_inference.model,
            deploy_params={
                "GPU": cfg.server_inference.gpu,
                "precision": cfg.model_inference.precision,
            },
        )

        validation_model_info = ModelInfo(
            url=cfg.server_validation.inference_url(),
            #url="http://0.0.0.0:8010",
            name=cfg.model_validation.model,
            deploy_params={
                "GPU": cfg.server_validation.gpu,
                "precision": cfg.model_validation.precision,
            },
        )
        os.makedirs(cfg.run_inference.output_path, exist_ok=True)
        request_params = cfg.run_inference.request
        setting_name = cfg.setting_filename()
        data_path = f"{cfg.run_inference.output_path}/{setting_name}"
        n_prompts = cfg.run_inference.n_prompts

        # Save run configuration next to results in a human-readable JSON
        config_filename = f"{setting_name.rsplit('.', 1)[0]}_config.json"
        config_path = f"{cfg.run_inference.output_path}/{config_filename}"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)

        prompts, languages = slice_mixed_language_prompts_with_langs(dataset, per_language_n=n_prompts//len(langs), langs=langs)
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


if __name__ == "__main__":
    main()