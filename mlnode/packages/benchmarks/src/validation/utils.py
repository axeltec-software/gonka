import json

import requests
import math
import threading
from typing import (
    Dict,
    Any,
    List,
    Callable,
    Optional,
    Tuple
)

from pydantic import BaseModel
import time
from requests.exceptions import Timeout, RequestException, ConnectionError
import difflib


from typing import Any, Dict, List
from pydantic import BaseModel, Field

from validation.data import (
    ModelInfo,
    RequestParams,
    ExperimentRequest,
    ValidationItem,
    Result,
    PositionResult
)

from common.logger import create_logger


logger = create_logger(__name__)

# A global registry of locks per output file path to serialize writes
_output_path_to_lock: Dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()

def _get_lock_for_path(path: str) -> threading.Lock:
    if not path:
        # No path provided; return a dummy lock that does nothing
        return threading.Lock()
    with _registry_lock:
        if path not in _output_path_to_lock:
            _output_path_to_lock[path] = threading.Lock()
        return _output_path_to_lock[path]

def _extract_logprob_token(s: str) -> str:
    if "token_id" in s:
        return s.split("token_id:")[1]
    return s

class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)

class EnforcedTokens(BaseModel):
    tokens: List[EnforcedToken]

    @classmethod
    def from_content(cls, content: List[Dict[str, Any]]) -> "EnforcedTokens":
        tokens = []
        for position in content:
            token = _extract_logprob_token(position["token"])
            top_tokens = [_extract_logprob_token(x["token"]) for x in position["top_logprobs"]]
            tokens.append(EnforcedToken(token=token, top_tokens=top_tokens))
        return cls(tokens=tokens)
    
    @classmethod
    def from_result(cls, result: Result) -> "EnforcedTokens":
        return cls(tokens=[EnforcedToken(token=r.token, top_tokens=list(r.logprobs.keys())) for r in result.results])

    
def _prepare_messages(
    prompt: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant. Response clear, correct and complete."},
        {"role": "user", "content": prompt}
    ]


def inference(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt: str,
) -> Dict[str, Any]:
    url = f"{model_info.url}/v1/chat/completions"
    payload = {
        "model": model_info.name,
        "messages": _prepare_messages(prompt),
        "max_tokens": request_params.max_tokens,
        "temperature": request_params.temperature,
        "seed": request_params.seed,
        "stream": False,
        "logprobs": True,
        "n": 1,
        "top_logprobs": request_params.top_logprobs,
        "skip_special_tokens": False,
        "repetition_penalty": 1.2,
        "return_tokens_as_token_ids": True,
    }
    
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Inference API request failed with status {response.status_code} {response.text}")
    return response.json()


def validation(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt: str,
    enforced_str: Optional[str] = None,
    enforced_tokens: Optional[EnforcedTokens] = None,
) -> Dict[str, Any]:
    url = f"{model_info.url}/v1/chat/completions"
    payload = {
        "model": model_info.name,
        "messages": _prepare_messages(prompt),
        "max_tokens": request_params.max_tokens,
        "temperature": request_params.temperature,
        "seed": request_params.seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": request_params.top_logprobs,
        #"prompt_logprobs": request_params.top_logprobs,
        "n": 1,
        "skip_special_tokens": False,
        "repetition_penalty": 1.2,
    }

    if enforced_str:
        payload["enforced_str"] = enforced_str
    if enforced_tokens:
        payload["enforced_tokens"] = enforced_tokens.dict()

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Validation API request failed with status {response.status_code} {response.text}\n(enforced_tokens: {enforced_tokens})\n(payload: {payload})")
    
    return response.json()



def _extract_logprobs(resp) -> Result:
    logprobs = resp["choices"][0]["logprobs"]["content"]
    text = resp["choices"][0]["message"]["content"]
    results = []
    for position in logprobs:
        res = PositionResult(
            token=_extract_logprob_token(position["token"]),
            logprobs={_extract_logprob_token(logprob["token"]): logprob["logprob"] for logprob in position["top_logprobs"]}
        )
        results.append(res)

    return Result(text=text, results=results)


def _extract_prompt_logprobs(resp, prompt_len) -> Result:
    logprobs_val = resp["prompt_logprobs"][prompt_len:]
    val_data = []
    token_ids = []
    for el in logprobs_val:
        top_logprobs = []
        for token_id, info in el.items():
            top_logprobs.append({
                    "token": token_id,
                    "logprob": info["logprob"],
                    "rank": info["rank"],
                    "decoded_token": info["decoded_token"]
                })

        pos = top_logprobs[0]
        token_ids.append(int(pos["token"]))
        pos["top_logprobs"] = top_logprobs
        val_data.append(pos)

    results = []
    for position in val_data:
        res = PositionResult(
            token=position["token"],
            logprobs={logprob["token"]: logprob["logprob"] for logprob in position["top_logprobs"]}
        )
        results.append(res)

    return Result(text="", results=results)


def _extract_enforced_tokens(resp) -> EnforcedTokens:
    return EnforcedTokens.from_content(resp["choices"][0]["logprobs"]["content"])


def find_text_differences(text1: str, text2: str) -> List[Tuple[str, int, int, str, int, int]]:
    """
    Find all differing substrings between two texts.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        List of tuples: (operation, start1, end1, start2, end2, content)
        where operation is 'replace', 'delete', or 'insert'
    """
    differences = []
    matcher = difflib.SequenceMatcher(None, text1, text2)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            differences.append(('replace', i1, i2, text1[i1:i2], j1, j2, text2[j1:j2]))
        elif tag == 'delete':
            differences.append(('delete', i1, i2, text1[i1:i2], j1, j2, ''))
        elif tag == 'insert':
            differences.append(('insert', i1, i2, '', j1, j2, text2[j1:j2]))
    
    return differences


def format_text_differences(text1: str, text2: str, label1: str = "Text 1", label2: str = "Text 2") -> str:
    """
    Format the differences between two texts in a human-readable way.
    
    Args:
        text1: First text string
        text2: Second text string
        label1: Label for the first text
        label2: Label for the second text
        
    Returns:
        Formatted string showing the differences
    """
    differences = find_text_differences(text1, text2)
    
    if not differences:
        return f"{label1} and {label2} are identical."
    
    output = [f"\nFound {len(differences)} difference(s) between {label1} and {label2}:\n"]
    output.append("="*80)
    
    for i, diff in enumerate(differences, 1):
        op, i1, i2, content1, j1, j2, content2 = diff
        
        output.append(f"\nDifference #{i} ({op}):")
        output.append(f"  Position in {label1}: [{i1}:{i2}]")
        output.append(f"  Position in {label2}: [{j1}:{j2}]")
        
        if op == 'replace':
            output.append(f"  {label1}: {repr(content1)}")
            output.append(f"  {label2}: {repr(content2)}")
        elif op == 'delete':
            output.append(f"  Deleted from {label1}: {repr(content1)}")
        elif op == 'insert':
            output.append(f"  Inserted in {label2}: {repr(content2)}")
        
        output.append("-"*40)
    
    return "\n".join(output)


def format_token_logprobs_differences(result1: Result, result2: Result, label1: str = "Result 1", label2: str = "Result 2") -> str:
    """
    Format the differences in tokens and logprobs between two results.
    
    Args:
        result1: First Result object with tokens and logprobs
        result2: Second Result object with tokens and logprobs
        label1: Label for the first result
        label2: Label for the second result
        
    Returns:
        Formatted string showing token and logprob differences
    """
    output = [f"\n{'='*80}\nToken-Level Comparison between {label1} and {label2}:\n{'='*80}\n"]
    
    tokens1 = [r.token for r in result1.results]
    tokens2 = [r.token for r in result2.results]
    
    output.append(f"Number of tokens in {label1}: {len(tokens1)}")
    output.append(f"Number of tokens in {label2}: {len(tokens2)}")
    output.append("")
    
    max_len = max(len(result1.results), len(result2.results))
    differences_found = 0
    
    for i in range(max_len):
        r1 = result1.results[i] if i < len(result1.results) else None
        r2 = result2.results[i] if i < len(result2.results) else None
        
        if r1 is None or r2 is None or r1.token != r2.token:
            differences_found += 1
            output.append(f"\n{'*'*80}")
            output.append(f"Position {i} - TOKENS DIFFER:")
            output.append(f"{'*'*80}")
            
            if r1:
                output.append(f"\n{label1} token: {repr(r1.token)}")
                output.append(f"  Logprobs (top {len(r1.logprobs)}):")
                for token, logprob in sorted(r1.logprobs.items(), key=lambda x: x[1], reverse=True)[:10]:
                    output.append(f"    {repr(token)}: {logprob:.6f}")
            else:
                output.append(f"\n{label1}: <missing>")
            
            if r2:
                output.append(f"\n{label2} token: {repr(r2.token)}")
                output.append(f"  Logprobs (top {len(r2.logprobs)}):")
                for token, logprob in sorted(r2.logprobs.items(), key=lambda x: x[1], reverse=True)[:10]:
                    output.append(f"    {repr(token)}: {logprob:.6f}")
            else:
                output.append(f"\n{label2}: <missing>")
            
            output.append(f"{'*'*80}")
        else:
            logprob_diff = abs(r1.logprobs.get(r1.token, float('-inf')) - r2.logprobs.get(r2.token, float('-inf')))
            
            top_tokens_1 = set(sorted(r1.logprobs.keys(), key=lambda x: r1.logprobs[x], reverse=True)[:5])
            top_tokens_2 = set(sorted(r2.logprobs.keys(), key=lambda x: r2.logprobs[x], reverse=True)[:5])
            
            if logprob_diff > 0.01 or top_tokens_1 != top_tokens_2:
                differences_found += 1
                output.append(f"\n{'-'*80}")
                output.append(f"Position {i} - Token: {repr(r1.token)} (SAME TOKEN, different logprobs)")
                output.append(f"{'-'*80}")
                
                output.append(f"\n{label1}:")
                output.append(f"  Token logprob: {r1.logprobs.get(r1.token, 'N/A'):.6f}")
                output.append(f"  Top tokens:")
                for token, logprob in sorted(r1.logprobs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    marker = " <--" if token == r1.token else ""
                    output.append(f"    {repr(token)}: {logprob:.6f}{marker}")
                
                output.append(f"\n{label2}:")
                output.append(f"  Token logprob: {r2.logprobs.get(r2.token, 'N/A'):.6f}")
                output.append(f"  Top tokens:")
                for token, logprob in sorted(r2.logprobs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    marker = " <--" if token == r2.token else ""
                    output.append(f"    {repr(token)}: {logprob:.6f}{marker}")
                
                output.append(f"  Logprob difference: {logprob_diff:.6f}")
                output.append(f"{'-'*80}")
    
    output.insert(3, f"Positions with differences: {differences_found}\n")
    
    return "\n".join(output)


def generate_and_validate(
    experiment_request: ExperimentRequest
) -> ValidationItem:
    # comment this part for validation with a single request
    # ================================
    inference_resp = inference(
         experiment_request.inference_model,
         experiment_request.request_params,
         experiment_request.prompt,
    )
    # try:
    #     with open(file="infer_resp.json", mode="w") as f:
    #         json.dump(inference_resp, f)
    #     exit(0)
    # except Exception as e:
    #     print(f"Failed to write to file: {e}")
    #     exit(1)
    # ================================

    # with open(file="infer_resp.json", mode="r") as f:
    #     inference_resp = json.load(f)

    inference_result = _extract_logprobs(inference_resp)
    enforced_tokens = _extract_enforced_tokens(inference_resp)

    validation_resp = validation(
        experiment_request.validation_model,
        experiment_request.request_params,
        experiment_request.prompt,
        # enforced_str=inference_result.text,
        enforced_tokens=enforced_tokens
    )

    # try:
    #     with open(file="val_resp.json", mode="w") as f:
    #         json.dump(validation_resp, f)
    #         #exit(0)
    # except Exception as e:
    #     print(f"Failed to write to file: {e}")
    #     exit(1)
    prompt_len = inference_resp["usage"]["prompt_tokens"]
    validation_result = _extract_prompt_logprobs(validation_resp, prompt_len)
    #validation_result = _extract_logprobs(validation_resp)

    #print(f"Validation text length = {len(validation_result.text)}, inference text length = {len(inference_result.text)}")
    #if validation_result.text != inference_result.text:
    inference_tok_ids = [el['token'] for el in inference_resp["choices"][0]["logprobs"]["content"]]
    validation_tok_ids = [list(el.keys())[0] for el in validation_resp["prompt_logprobs"][prompt_len:]]
    if inference_tok_ids != validation_tok_ids:
        raise RuntimeError(
            f"token id sequences don't match\n" +
            f"inference:\n {inference_tok_ids}\n" +
            f"{'-'*10}\n" +
            f"validation:\n {validation_tok_ids}\n" +
            f"{'-'*100}"
            )
    
        # diff_report = format_text_differences(
        #     inference_result.text, 
        #     validation_result.text,
        #     label1="inference",
        #     label2="validation"
        # )
        
        # token_diff_report = format_token_logprobs_differences(
        #     inference_result,
        #     validation_result,
        #     label1="inference",
        #     label2="validation"
        # )

        # if experiment_request.output_path:
        #     diff_file_path = experiment_request.output_path.replace('.jsonl', '_diff.txt')
        #     lock = _get_lock_for_path(diff_file_path)
        #     with lock:
        #         try:
        #             with open(diff_file_path, 'a') as f:
        #                 f.write(f"\n{'='*100}\n")
        #                 f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        #                 f.write(f"Prompt: {experiment_request.prompt[:100]}...\n")
        #                 f.write(f"\n{diff_report}\n")
        #                 f.write(f"\n{token_diff_report}\n")
        #                 f.write(f"\nFull inference text: {repr(inference_result.text)}\n")
        #                 f.write(f"Full validation text: {repr(validation_result.text)}\n")
        #                 f.write(f"{'='*100}\n\n")
        #             logger.info(f"Diff report saved to {diff_file_path}")
        #         except Exception as e:
        #             logger.error(f"Failed to write diff report to {diff_file_path}: {e}")
        
        # raise RuntimeError(
        #     f"Text sequences don't match between inference and validation.\n"
        #     #f"{diff_report}\n"
        #     #f"{token_diff_report}\n"
        #     f"Full inference text: {repr(inference_result.text)}\n"
        #     f"Full validation text: {repr(validation_result.text)}\n"
        #     f"Inference result: {[el['token'] for el in inference_resp["choices"][0]["logprobs"]["content"]]}\n"
        #     f"Validation result: {[list(el.keys())[0] for el in validation_resp["prompt_logprobs"][prompt_len:]]}\n"
        # )

    item = experiment_request.to_result(
        inference_result,
        validation_result
    )
    if experiment_request.output_path:
        lock = _get_lock_for_path(experiment_request.output_path)
        with lock:
            try:
                with open(experiment_request.output_path, 'a') as f:
                    f.write(item.model_dump_json() + '\n')
            except Exception as e:
                logger.error(f"Failed to write result to {experiment_request.output_path}: {e}")

    return item


def token_distance(
    inf_position_logprobs: PositionResult,
    val_position_logprobs: PositionResult
):
    dist = 0
    n_matches = 0
    for k, v in inf_position_logprobs.logprobs.items():
        if k in val_position_logprobs.logprobs:
            n_matches += 1
            dist += abs(v - val_position_logprobs.logprobs[k]) / (1e-10 + abs(v) + abs(val_position_logprobs.logprobs[k])) / 2.
    return dist, n_matches



def _check_match(
    inf_result: Result,
    val_result: Result,
):
    if [r.token for r in inf_result.results] != [r.token for r in val_result.results]:
        logger.debug(
            f"tokens sequences don't match\n" +
            f"inference:\n {[r.token for r in inf_result.results]}\n" +
            f"{'-'*10}\n" +
            f"validation:\n {[r.token for r in val_result.results]}\n" +
            f"{'-'*100}"
        )
        return False
    return True

def distance(
    inf_result: Result,
    val_result: Result,
    distance_func: Callable = token_distance
):

    if not _check_match(inf_result, val_result):
        return -1, -1

    total_dist = 0
    total_n_matches = 0
    for inf_position, val_position in zip(inf_result.results, val_result.results):
        dist, n_matches = distance_func(inf_position, val_position)
        total_dist += dist
        total_n_matches += n_matches
    
    matches_ratio = total_n_matches / (len(inf_result.results)*len(inf_result.results[0].logprobs))
    total_dist /= (len(inf_result.results)*len(inf_result.results[0].logprobs))
    return total_dist, matches_ratio


def token_distance2(
    inf_position_logprobs: PositionResult,
    val_position_logprobs: PositionResult
):
    dist = 0.0
    n_matches = 0

    if not val_position_logprobs.logprobs:
        return len(inf_position_logprobs.logprobs), 0

    sorted_logprobs = sorted(val_position_logprobs.logprobs.values())
    
    if len(sorted_logprobs) >= 2:
        min_val_logprob_1 = sorted_logprobs[0]
        min_val_logprob_2 = sorted_logprobs[1]
    else:
        min_val_logprob_1 = sorted_logprobs[0]
        min_val_logprob_2 = min_val_logprob_1 - 1.0

    for token, inf_logprob in inf_position_logprobs.logprobs.items():
        if token in val_position_logprobs.logprobs:
            val_logprob = val_position_logprobs.logprobs[token]
            n_matches += 1
        else:
            val_logprob = min_val_logprob_1 - (min_val_logprob_2 - min_val_logprob_1)

        denom = 1e-10 + abs(inf_logprob) + abs(val_logprob)
        dist += abs(inf_logprob - val_logprob) / denom / 2.0

    return dist, n_matches


def similarity2(
    inf_result: Result,
    val_result: Result,
):
    dist, matches_ratio = distance2(inf_result, val_result)
    if dist == -1:
        return -1, -1
    return 1 - dist, matches_ratio


def distance2(inf_result: Result, val_result: Result):
    if not _check_match(inf_result, val_result):
        return -1, -1

    total_dist = 0
    total_n_matches = 0
    for inf_position, val_position in zip(inf_result.results, val_result.results):
        dist, n_matches = token_distance2(inf_position, val_position)
        total_dist += dist
        total_n_matches += n_matches
    
    matches_ratio = total_n_matches / (len(inf_result.results)*len(inf_result.results[0].logprobs))
    total_dist = (total_dist + 1.0) / (max(100, len(inf_result.results))*len(inf_result.results[0].logprobs) + 1.0)
    return total_dist, matches_ratio



import numpy as np
from typing import List, Dict
from validation.data import Result

BAD_LOGP = -10.0

def _clean_logprob(lp: float, floor: float = BAD_LOGP) -> float:
    return lp if lp is not None and lp > floor else floor


def get_metric(logprobs: List[float]) -> float:
    if not logprobs:
        return 0.0
    return float(np.exp(np.mean(logprobs)))


def get_metric_from_result(inf_result: Result) -> float:
    per_token_lp: List[float] = []

    for r in inf_result.results:
        lp = r.logprobs.get(r.token, BAD_LOGP)
        per_token_lp.append(_clean_logprob(lp))

    return get_metric(per_token_lp)
