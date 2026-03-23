from pyarrow import int32
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
    InferenceSnapshot,
    Result,
    PositionResult,
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
    # tokens: List[EnforcedToken]
    tokens: List[int]

    @classmethod
    def from_content(cls, content: List[Dict[str, Any]]) -> "EnforcedTokens":
        tokens = []
        for position in content:
            token = _extract_logprob_token(position["token"])
            # top_tokens = [_extract_logprob_token(x["token"]) for x in position["top_logprobs"]]
            # tokens.append(EnforcedToken(token=token, top_tokens=top_tokens))
            tokens.append(int(token))
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
        "return_token_ids": True, #return prompt token ids
        "return_tokens_as_token_ids": True
    }
    
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Inference API request failed with status {response.status_code} {response.text}")
    return response.json()


def validation(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt_tokens: List[int],
    enforced_tokens: List[int],
) -> Dict[str, Any]:
    url = f"{model_info.url}/v1/completions"
    payload = {
        "model": model_info.name,
        "prompt": prompt_tokens + enforced_tokens,
        "max_tokens": 1,
        "temperature": request_params.temperature,
        "seed": request_params.seed,
        "stream": False,
        "repetition_penalty": 1.2,
        "prompt_logprobs": request_params.top_logprobs,
        "return_tokens_as_token_ids": True
    }

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


def _extract_enforced_tokens(resp) -> EnforcedTokens:
    return EnforcedTokens.from_content(resp["choices"][0]["logprobs"]["content"])


def _extract_prompt_logprobs(resp, prompt_len) -> Result:
    logprobs_val = resp["choices"][0]["prompt_logprobs"][prompt_len:]
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

def generate_and_validate(
    experiment_request: ExperimentRequest
) -> ValidationItem:
    inference_resp = inference(
        experiment_request.inference_model,
        experiment_request.request_params,
        experiment_request.prompt,
    )

    inference_prompt_len = inference_resp["usage"]["prompt_tokens"]
    inference_prompt_tokens = inference_resp["prompt_token_ids"]
    inference_result = _extract_logprobs(inference_resp)
    enforced_tokens = _extract_enforced_tokens(inference_resp)

    validation_resp = validation(
        experiment_request.validation_model,
        experiment_request.request_params,
        prompt_tokens=inference_prompt_tokens,
        enforced_tokens=enforced_tokens.tokens
    )
    validation_result = _extract_prompt_logprobs(validation_resp, inference_prompt_len)
    inference_tok_ids = [
        _extract_logprob_token(el['token']) 
        for el in inference_resp["choices"][0]["logprobs"]["content"]
    ]
    validation_tok_ids = [
        list(el.keys())[0] 
        for el in validation_resp["choices"][0]["prompt_logprobs"][inference_prompt_len:]
    ]

    if inference_tok_ids != validation_tok_ids:
        print(
            f"token id sequences don't match\n" +
            f"inference:\n {inference_tok_ids}\n" +
            f"{'-'*10}\n" +
            f"validation:\n {validation_tok_ids}\n" +
            f"{'-'*100}"
        )


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


def generate_inference_snapshot(
    experiment_request: ExperimentRequest,
    output_path: Optional[str] = None,
) -> InferenceSnapshot:
    """Run inference only and return a snapshot that can be used for validation later."""
    inference_resp = inference(
        experiment_request.inference_model,
        experiment_request.request_params,
        experiment_request.prompt,
    )

    inference_prompt_len = inference_resp["usage"]["prompt_tokens"]
    inference_prompt_tokens = inference_resp["prompt_token_ids"]
    inference_result = _extract_logprobs(inference_resp)
    enforced_tokens = _extract_enforced_tokens(inference_resp)

    snapshot = InferenceSnapshot(
        prompt=experiment_request.prompt,
        language=experiment_request.language,
        inference_result=inference_result,
        inference_prompt_tokens=inference_prompt_tokens,
        inference_prompt_len=inference_prompt_len,
        enforced_tokens=enforced_tokens.tokens,
        inference_model=experiment_request.inference_model,
        validation_model=experiment_request.validation_model,
        request_params=experiment_request.request_params,
    )

    if output_path:
        lock = _get_lock_for_path(output_path)
        with lock:
            try:
                with open(output_path, 'a') as f:
                    f.write(snapshot.model_dump_json() + '\n')
            except Exception as e:
                logger.error(f"Failed to write snapshot to {output_path}: {e}")

    return snapshot


def validate_from_snapshot(
    snapshot: InferenceSnapshot,
    output_path: Optional[str] = None,
) -> ValidationItem:
    """Run validation using a previously saved inference snapshot."""
    validation_resp = validation(
        snapshot.validation_model,
        snapshot.request_params,
        prompt_tokens=snapshot.inference_prompt_tokens,
        enforced_tokens=snapshot.enforced_tokens,
    )
    validation_result = _extract_prompt_logprobs(validation_resp, snapshot.inference_prompt_len)

    inference_tok_ids = [str(t) for t in snapshot.enforced_tokens]
    validation_tok_ids = [
        list(el.keys())[0]
        for el in validation_resp["choices"][0]["prompt_logprobs"][snapshot.inference_prompt_len:]
    ]
    if inference_tok_ids != validation_tok_ids:
        print(
            f"token id sequences don't match\n"
            f"inference:\n {inference_tok_ids}\n"
            f"{'-'*10}\n"
            f"validation:\n {validation_tok_ids}\n"
            f"{'-'*100}"
        )

    item = ValidationItem(
        prompt=snapshot.prompt,
        language=snapshot.language,
        inference_result=snapshot.inference_result,
        validation_result=validation_result,
        inference_model=snapshot.inference_model,
        validation_model=snapshot.validation_model,
        request_params=snapshot.request_params,
    )

    if output_path:
        lock = _get_lock_for_path(output_path)
        with lock:
            try:
                with open(output_path, 'a') as f:
                    f.write(item.model_dump_json() + '\n')
            except Exception as e:
                logger.error(f"Failed to write result to {output_path}: {e}")

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
