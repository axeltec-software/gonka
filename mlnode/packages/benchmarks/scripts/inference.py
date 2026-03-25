import sys
sys.path.append('../src')
sys.path.append('../../common/src')

import os
import copy
import json
import argparse
import time
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
from transformers import AutoTokenizer
from tqdm import tqdm

from validation.runner import run_validation
from validation.prompts import preload_all_language_prompts, slice_mixed_language_prompts_with_langs
from validation.data import *
from validation.utils import VLLMRequestWrapper, VLLMRequestWrapper_AppendEnforcedTokensInVLLM, VLLMRequestWrapper_AppendEnforcedTokensInPayload, _extract_prompt_logprobs
from common.logger import create_logger
from validation.model_presets import QWEN3_600M_FP8, QWEN3_600M_FP16, QWQ_32B_FP8, QWQ_32B_INT4, QWEN3_30B_FP8, QWEN3_30B_INT4

logger = create_logger(__name__)
    
N_PROMPTS = 100
MAX_WORKERS = None
CONNECTION_ERROR_WAIT_TIME = 120  # seconds to wait after connection error
run_params_high_temp = RunParams(
    exp_name='qwen3-600M',
    output_path='data/inference_results',
    n_prompts=N_PROMPTS,
    timeout=1800,
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
    ip='86.38.238.60',
    inference_port='8005',
    node_port='17340',
    gpu='1xH100',
)

server_1xH100_2 = ServerConfig(
    ip='86.38.238.60',
    inference_port='8010',
    node_port='19145',
    gpu='1xH100',
)

server_4x3090_1 = ServerConfig(
    ip='localhost',
    inference_port='8000',
    node_port='24434',
    gpu='2x3090',
)

server_4x3090_2 = ServerConfig(
    ip='localhost',
    inference_port='8080',
    node_port='20505',
    gpu='2x3090',
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

server_1xRTX4000_1 = ServerConfig(
    ip='0.0.0.0',
    inference_port='8005',
    node_port='17340',
    gpu='1xRTX4000',
)

server_1xRTX4000_2 = ServerConfig(
    ip='0.0.0.0',
    inference_port='8010',
    node_port='19145',
    gpu='1xRTX4000',
)

honest_preset = QWEN3_600M_FP16
fraudulent_preset = QWEN3_600M_FP8

langs = ("en", "sp","ch", "hi", "ar")
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


def inference_with_retry(req_wrapper: VLLMRequestWrapper, model_info: ModelInfo, request_params: RequestParams, prompt: Prompt, 
                        max_retries: int = 3, wait_time: int = CONNECTION_ERROR_WAIT_TIME):
    """Wrapper for inference with retry logic and connection error handling"""
    for attempt in range(max_retries):
        try:
            return req_wrapper.inference(model_info, request_params, prompt)
        except (ConnectionError, ConnectionRefusedError, RequestException) as e:
            if "Connection refused" in str(e) or "Max retries exceeded" in str(e):
                logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts")
                    raise
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error during inference: {e}")
            raise
    raise RuntimeError(f"Failed to complete inference after {max_retries} attempts")


def validation_with_retry(req_wrapper: VLLMRequestWrapper, model_info: ModelInfo, request_params: RequestParams, prompt: Prompt, 
                         enforced_tokens=None, max_retries: int = 3, wait_time: int = CONNECTION_ERROR_WAIT_TIME):
    """Wrapper for validation with retry logic and connection error handling"""
    for attempt in range(max_retries):
        try:
            return req_wrapper.validation(model_info, request_params, prompt, enforced_tokens=enforced_tokens)
        except (ConnectionError, ConnectionRefusedError, RequestException) as e:
            if "Connection refused" in str(e) or "Max retries exceeded" in str(e):
                logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts")
                    raise
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            raise
    raise RuntimeError(f"Failed to complete validation after {max_retries} attempts")


class InferenceRequest:
    """Stores a single inference request"""
    def __init__(self, prompt: Prompt, language: Optional[str], idx: int):
        self.prompt = prompt
        self.language = language
        self.idx = idx


class InferenceResponse:
    """Stores the response from inference server"""
    def __init__(self, prompt: Prompt, language: Optional[str], idx: int, prompt_len: int, 
                 inference_text: str, inference_result: dict, enforced_tokens: dict):
        self.prompt = prompt
        self.language = language
        self.idx = idx
        self.prompt_len=prompt_len
        self.inference_text = inference_text
        self.inference_result = inference_result
        self.enforced_tokens = enforced_tokens
    
    def to_dict(self):
        return {
            'prompt': json.loads(self.prompt.model_dump_json()),
            'language': self.language,
            'idx': self.idx,
            'prompt_len': self.prompt_len,
            'inference_text': self.inference_text,
            'inference_result': self.inference_result,
            'enforced_tokens': self.enforced_tokens
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            prompt=Prompt.model_validate_json(json.dumps(data['prompt'])),
            language=data.get('language'),
            idx=data['idx'],
            prompt_len=data['prompt_len'],
            inference_text=data['inference_text'],
            inference_result=data['inference_result'],
            enforced_tokens=data['enforced_tokens']
        )


def run_inference_only(
    req_wrapper: VLLMRequestWrapper, 
    prompts: List[str],
    languages: List[str],
    inference_model: ModelInfo,
    request_params: RequestParams,
    max_workers: int,
    output_path: str,
    max_retries: int = 3,
    wait_time: int = CONNECTION_ERROR_WAIT_TIME
):
    """Run inference and save responses to file"""
    logger.info(f"Running inference on {len(prompts)} prompts")
    
    inference_requests = [
        InferenceRequest(prompt=Prompt.from_string(prompts[i]), language=languages[i], idx=i)
        for i in range(len(prompts))
    ]
    
    results = []
    
    def process_inference(req: InferenceRequest):
        try:
            inference_resp = inference_with_retry(
                req_wrapper,
                inference_model,
                request_params,
                req.prompt,
                max_retries=max_retries,
                wait_time=wait_time
            )
            inference_result = req_wrapper._extract_logprobs(inference_resp)
            enforced_tokens = req_wrapper._extract_enforced_tokens(inference_resp)
            prompt_len = int(inference_resp["usage"]["prompt_tokens"])
            

            req.prompt.set_tokens(inference_resp["prompt_token_ids"])

            res = InferenceResponse(
                prompt=req.prompt,
                language=req.language,
                idx=req.idx,
                prompt_len=prompt_len,
                inference_text=inference_result.text,
                inference_result=inference_result.model_dump(),
                enforced_tokens=enforced_tokens.model_dump()
            )
            return res

        except Exception as e:
            logger.exception(f"Failed to process inference for prompt {req.idx}: {e}")
            raise
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_inference, req): req for req in inference_requests}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running inference", leave=False):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
    
    # Sort by index to maintain order
    results.sort(key=lambda x: x.idx)
    
    # Save to file
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')
    
    logger.info(f"Saved {len(results)} inference responses to {output_path}")
    return results


def run_validation_only(
    req_wrapper: VLLMRequestWrapper,
    inference_responses_path: str,
    validation_model: ModelInfo,
    request_params: RequestParams,
    max_workers: int,
    output_path: str,
    inference_model_info: ModelInfo,
    max_retries: int = 3,
    wait_time: int = CONNECTION_ERROR_WAIT_TIME
):
    """Load inference responses and run validation"""
    logger.info(f"Loading inference responses from {inference_responses_path}")
    
    # Load inference responses
    inference_responses = []
    with open(inference_responses_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            inference_responses.append(InferenceResponse.from_dict(data))
    
    logger.info(f"Loaded {len(inference_responses)} inference responses")
    logger.info(f"Running validation on {len(inference_responses)} responses")
    
    def process_validation(inf_resp: InferenceResponse):
        try:
            # Reconstruct enforced tokens
            enforced_tokens = EnforcedTokens.model_validate(inf_resp.enforced_tokens)
            
            # Run validation with retry
            validation_resp = validation_with_retry(
                req_wrapper,
                validation_model,
                request_params,
                inf_resp.prompt,
                enforced_tokens=enforced_tokens,
                max_retries=max_retries,
                wait_time=wait_time
            )
            prompt_len = inf_resp.prompt_len

            logprobs_val = req_wrapper.validation_response_to_logprobs_val(validation_resp, prompt_len)
            validation_result = _extract_prompt_logprobs(logprobs_val)
            
            # Reconstruct inference result
            inference_result = Result.model_validate(inf_resp.inference_result)

            inference_tok_ids = [el['token'] for el in inf_resp.enforced_tokens["tokens"]]
            validation_tok_ids = [list(el.keys())[0] for el in req_wrapper.validation_response_to_logprobs_val(validation_resp, prompt_len)]
            
            # Check if token ids match
            if inference_tok_ids != validation_tok_ids:
                raise RuntimeError(
                    f"token id sequences don't match\n" +
                    f"inference:\n {inference_tok_ids}\n" +
                    f"{'-'*10}\n" +
                    f"validation:\n {validation_tok_ids}\n" +
                    f"{'-'*100}"
                )
            
            # Create ValidationItem
            item = ValidationItem(
                prompt=inf_resp.prompt,
                language=inf_resp.language,
                inference_result=inference_result,
                validation_result=validation_result,
                inference_model=inference_model_info,
                validation_model=validation_model,
                request_params=request_params
            )
            
            return item
        except Exception as e:
            logger.exception(f"Failed to process validation for prompt {inf_resp.idx}: {e}")
            raise
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_validation, resp): resp for resp in inference_responses}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running validation", leave=False):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
    
    # Save to file
    with open(output_path, 'w') as f:
        for result in results:
            f.write(result.model_dump_json() + '\n')
    
    logger.info(f"Saved {len(results)} validation results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run inference and validation separately or together'
    )
    parser.add_argument('--enforced-tokens-in', type=str, choices=['vllm', 'payload'], default='payload', 
                        help="Using a vLLM version customized for prefill validation ('vllm') or an original vLLM version ('payload' -- by default)")
    parser.add_argument('--type', type=str, choices=['sequential', 'parallel'], default='parallel', 
                        help="Inference-validation workflow type: parallel - for simultaneous runs of inference and validation (if executor and validator are running in parallel), " \
                        "sequential - for running inference and validation sequentially (if executor and validator can't be run simultaneously)")

    parser.add_argument(
        '--mode',
        type=str,
        choices=['none', 'inference', 'validation', 'both'],
        default='none',
        help="Mode to run with sequential execution: inference only, validation only, or both. For sequential this is mandatory. For parallel this is not applicable and has to be 'none'"
    )
    parser.add_argument(
        '--inference-responses',
        type=str,
        help='Path to inference responses file (required for validation mode)'
    )
    parser.add_argument(
        '--connection-error-wait',
        type=int,
        default=CONNECTION_ERROR_WAIT_TIME,
        help=f'Seconds to wait after connection error before retry (default: {CONNECTION_ERROR_WAIT_TIME})'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum number of retries for connection errors (default: 3)'
    )
    
    args = parser.parse_args()
    
    if args.type == 'parallel':
        if args.mode != 'none':
            parser.error("--type=parallel is incompatible with option --mode")

        if args.inference_responses:
            parser.error("--type=parallel is incompatible with option --inference-responses")

        if args.connection_error_wait != CONNECTION_ERROR_WAIT_TIME:
            parser.error("--type=parallel is incompatible with option --connection-error-wait")

    if args.type == 'sequential' and args.mode == 'none':
        parser.error("Missing --mode for --type=parallel")

    if args.mode == 'validation' and not args.inference_responses:
        parser.error("--inference-responses is required when mode is 'validation'")
    
    dataset = preload_all_language_prompts(langs=langs)
    
    for cfg in runs:
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
        
        os.makedirs(cfg.run_inference.output_path, exist_ok=True)
        request_params = cfg.run_inference.request
        setting_name = cfg.setting_filename()
        data_path = f"{cfg.run_inference.output_path}/{setting_name}"
        n_prompts = cfg.run_inference.n_prompts
        
        # Save run configuration
        config_filename = f"{setting_name.rsplit('.', 1)[0]}_config.json"
        config_path = f"{cfg.run_inference.output_path}/{config_filename}"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)
        
        prompts, languages = slice_mixed_language_prompts_with_langs(
            dataset, per_language_n=n_prompts//len(langs), langs=langs
        )

        if args.enforced_tokens_in == 'vllm':
            req_wrapper = VLLMRequestWrapper_AppendEnforcedTokensInVLLM()
        else:
            req_wrapper = VLLMRequestWrapper_AppendEnforcedTokensInPayload()


        if args.type == 'parallel':
            if args.max_retries != 3: request_params.retries_max_attempts = args.max_retries
            _ = run_validation(
            req_wrapper,
            prompts,
            languages=languages,
            inference_model=inference_model_info,
            validation_model=validation_model_info,
            request_params=request_params,
            max_workers=cfg.max_workers,
            output_path=data_path,
            )
            print(f"Completed run. Results saved to: {data_path}")

        else:
            max_workers = cfg.max_workers if cfg.max_workers else 10
            if args.mode == 'inference':
                # Run inference only
                inference_responses_path = f"{cfg.run_inference.output_path}/{setting_name.rsplit('.', 1)[0]}_inference_responses.jsonl"
                run_inference_only(
                    req_wrapper=req_wrapper,
                    prompts=prompts,
                    languages=languages,
                    inference_model=inference_model_info,
                    request_params=request_params,
                    max_workers=max_workers,
                    output_path=inference_responses_path,
                    max_retries=args.max_retries,
                    wait_time=args.connection_error_wait
                )
                print(f"Completed inference. Responses saved to: {inference_responses_path}")
                print(f"To run validation, use: --mode validation --inference-responses {inference_responses_path}")
                
            elif args.mode == 'validation':
                # Run validation only
                run_validation_only(
                    req_wrapper=req_wrapper,
                    inference_responses_path=args.inference_responses,
                    validation_model=validation_model_info,
                    request_params=request_params,
                    max_workers=max_workers,
                    output_path=data_path,
                    inference_model_info=inference_model_info,
                    max_retries=args.max_retries,
                    wait_time=args.connection_error_wait
                )
                print(f"Completed validation. Results saved to: {data_path}")
                
            else:  # both
                # Run inference first
                inference_responses_path = f"{cfg.run_inference.output_path}/{setting_name.rsplit('.', 1)[0]}_inference_responses.jsonl"
                run_inference_only(
                    req_wrapper=req_wrapper,
                    prompts=prompts,
                    languages=languages,
                    inference_model=inference_model_info,
                    request_params=request_params,
                    max_workers=max_workers,
                    output_path=inference_responses_path,
                    max_retries=args.max_retries,
                    wait_time=args.connection_error_wait
                )
                print(f"Completed inference. Responses saved to: {inference_responses_path}")
                
                # Then run validation
                run_validation_only(
                    req_wrapper=req_wrapper,
                    inference_responses_path=inference_responses_path,
                    validation_model=validation_model_info,
                    request_params=request_params,
                    max_workers=max_workers,
                    output_path=data_path,
                    inference_model_info=inference_model_info,
                    max_retries=args.max_retries,
                    wait_time=args.connection_error_wait
                )
                print(f"Completed validation. Results saved to: {data_path}")


if __name__ == "__main__":
    main()

