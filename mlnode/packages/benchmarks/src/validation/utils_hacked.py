import requests
import math
import threading
from typing import (
    Dict,
    Any,
    List,
    Callable,
    Optional
)

from pydantic import BaseModel


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

_output_path_to_lock: Dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()

def _get_lock_for_path(path: str) -> threading.Lock:
    if not path:
        return threading.Lock()
    with _registry_lock:
        if path not in _output_path_to_lock:
            _output_path_to_lock[path] = threading.Lock()
        return _output_path_to_lock[path]


class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)

class EnforcedTokens(BaseModel):
    tokens: List[EnforcedToken]

    @classmethod
    def from_content(cls, content: List[Dict[str, Any]]) -> "EnforcedTokens":
        tokens = []
        for position in content:
            token = position["token"]
            top_tokens = [x["token"] for x in position["top_logprobs"]]
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


def _sampling_extras(request_params: RequestParams) -> Dict[str, Any]:
    """Return optional sampling params that are set (non-None) plus additional_params."""
    extras: Dict[str, Any] = {}
    if request_params.top_p is not None:
        extras["top_p"] = request_params.top_p
    if request_params.top_k is not None:
        extras["top_k"] = request_params.top_k
    if request_params.repetition_penalty is not None:
        extras["repetition_penalty"] = request_params.repetition_penalty
    extras.update(request_params.additional_params)
    return extras


def inference(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt: str,
) -> Dict[str, Any]:
    url = f"{model_info.url.rstrip('/')}/v1/chat/completions"
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
        **_sampling_extras(request_params),
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
    url = f"{model_info.url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_info.name,
        "messages": _prepare_messages(prompt),
        "max_tokens": request_params.max_tokens,
        "temperature": request_params.temperature,
        "seed": request_params.seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": request_params.top_logprobs,
        "n": 1,
        "skip_special_tokens": False,
        **_sampling_extras(request_params),
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
            token=position["token"],
            logprobs={logprob["token"]: logprob["logprob"] for logprob in position["top_logprobs"]}
        )
        results.append(res)

    return Result(text=text, results=results)


def inference_prefill(
    model_info: ModelInfo,
    request_params: RequestParams,
    prompt: str,
) -> Dict[str, Any]:
    """Score prompt tokens via /v1/chat/completions with a prefilled assistant message.

    Splits prompt at '<think>': everything before becomes the user message,
    everything from '<think>' onwards becomes the prefilled assistant content.
    Uses prompt_logprobs so the server returns per-position logprobs for every
    token in the formatted prompt (including the inference-result tokens).
    This ensures '<think>' is tokenised as its special token (151667), not split
    into character-level tokens as it would be in the raw completions endpoint.
    """
    think_pos = prompt.find("<think>")
    user_content = prompt[:think_pos] if think_pos != -1 else prompt
    assistant_content = prompt[think_pos:] if think_pos != -1 else ""

    url = f"{model_info.url.rstrip('/')}/v1/chat/completions"
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant. Response clear, correct and complete."},
        {"role": "user", "content": user_content},
    ]
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})

    payload: Dict[str, Any] = {
        "model": model_info.name,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0.0, #request_params.temperature,
        "top_p": 1.0, #request_params.top_p,
        "top_k": -1, #request_params.top_k,
        "seed": request_params.seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": request_params.top_logprobs,
        "n": 1,
        "skip_special_tokens": False,
        "add_generation_prompt": False,
        "prompt_logprobs": request_params.top_logprobs,
        "return_token_ids": True
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"Prefill inference request failed with status {response.status_code}: {response.text[:2000]}"
        )
    resp = response.json()
    if "choices" not in resp:
        raise RuntimeError(
            f"/v1/chat/completions returned 200 but missing 'choices'. Response: {response.text[:2000]}"
        )
    if "prompt_logprobs" not in resp:
        raise RuntimeError(
            f"/v1/chat/completions returned 200 but missing 'prompt_logprobs'. "
            f"Keys present: {list(resp.keys())}. Response: {str(resp)[:2000]}"
        )
    return resp


def _extract_prefill_logprobs(
    resp: Dict[str, Any],
    prompt: str,
    text_start: int = 0,
    #include_token_str: bool = True,
) -> Result:
    """Build Results from prompt_logprobs, starting from the '<think>' token (151667)."""
    import sys

    prompt_lps: List[Optional[Dict]] = resp["prompt_logprobs"]
    # prompt_lps_orig = prompt_lps
    # from collections import OrderedDict
    # for tok in prompt_lps:
    #     if tok is not None and isinstance(tok, dict):
    #         tok_ord = OrderedDict(tok)
    #         tok = tok_ord

    _THINK_TOKEN = "151667"

    # def first(s):
    #     '''Return the first element from an ordered collection
    #     or an arbitrary element from an unordered collection.
    #     Raise StopIteration if the collection is empty.
    #     '''
    #     return next(iter(s))
    
    prompt_ids = resp["prompt_token_ids"]

    # start_idx = next(
    #     (
    #         i for i, entry in enumerate(prompt_lps)
    #         if isinstance(entry, dict) and _THINK_TOKEN in first(entry)
    #     ),
    #     None,
    # )

    for tok in prompt_ids:
        if tok == int(_THINK_TOKEN):
            start_idx = prompt_ids.index(tok)
            break

    if start_idx is None:
        non_null = [e for e in prompt_lps if e is not None]
        print(
            f"[DEBUG] <think> token ({_THINK_TOKEN}) not found in prompt_logprobs. "
            f"Total entries: {len(prompt_lps)}, non-null: {len(non_null)}. "
            f"First 3 non-null: {non_null[:3]}",
            file=sys.stderr,
        )
        return Result(text=prompt[text_start:], results=[])

    _STOP_TOKEN_IDS = {"151645", "151643"}  # <|im_end|>, <|endoftext|>

    results = []
    decoded_parts: List[str] = []
    for j, entry in enumerate(prompt_lps[start_idx:]):
        actual_idx = start_idx + j
        if actual_idx >= len(prompt_ids):
            break
        actual_token = str(prompt_ids[actual_idx])
        if not isinstance(entry, dict):
            results.append(PositionResult(token=actual_token, logprobs={}))
            if actual_token in _STOP_TOKEN_IDS:
                break
            continue
        chosen_decoded: Optional[str] = None
        lp_map: Dict[str, float] = {}
        for tok_str, info in entry.items():
            if isinstance(info, dict):
                lp = info.get("logprob", 0.0)
                lp_map[tok_str] = lp
                if tok_str == actual_token:
                    chosen_decoded = info.get("decoded_token")
            elif isinstance(info, (int, float)):
                lp_map[tok_str] = float(info)
        # if chosen_decoded is not None:
        #     print(
        #     f"chosen_decoded: {chosen_decoded}, actual_token: {actual_token}",
        #     file=sys.stderr,
        # )
        results.append(PositionResult(
            token=actual_token,
            logprobs=lp_map,
            #token_str=chosen_decoded if include_token_str else None,
        ))
        if actual_token in _STOP_TOKEN_IDS:
            break
        if chosen_decoded is not None:
            decoded_parts.append(chosen_decoded)

    text = "".join(decoded_parts) if decoded_parts else prompt[text_start:]
    return Result(text=text, results=results)


def _extract_enforced_tokens(resp) -> EnforcedTokens:
    return EnforcedTokens.from_content(resp["choices"][0]["logprobs"]["content"])


def generate_and_validate(
    experiment_request: ExperimentRequest
) -> ValidationItem:
    inference_resp = inference(
        experiment_request.inference_model,
        experiment_request.request_params,
        experiment_request.prompt,
    )
    inference_result = _extract_logprobs(inference_resp)
    enforced_tokens = _extract_enforced_tokens(inference_resp)
    validation_resp = validation(
        experiment_request.validation_model,
        experiment_request.request_params,
        experiment_request.prompt,
        enforced_tokens=enforced_tokens
    )
    validation_result = _extract_logprobs(validation_resp)
    if validation_result.text != inference_result.text:
        raise RuntimeError(
            "Text sequences don't match between inference and validation."
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
    val_position_logprobs: PositionResult,
):
    """Matches Go customDistance/positionDistance: iterates over validation
    tokens, builds fallback from inference (original) side."""
    dist = 0.0
    n_matches = 0

    if not inf_position_logprobs.logprobs or not val_position_logprobs.logprobs:
        return 100.0, 0

    sorted_inf_logprobs = sorted(inf_position_logprobs.logprobs.values())

    if len(sorted_inf_logprobs) >= 2:
        min_inf_1 = sorted_inf_logprobs[0]
        min_inf_2 = sorted_inf_logprobs[1]
    else:
        min_inf_1 = sorted_inf_logprobs[0]
        min_inf_2 = min_inf_1 - 100.0

    next_inf_logprob = min_inf_1 - (min_inf_2 - min_inf_1)

    for token, val_logprob in val_position_logprobs.logprobs.items():
        if token in inf_position_logprobs.logprobs:
            inf_logprob = inf_position_logprobs.logprobs[token]
            n_matches += 1
        else:
            inf_logprob = next_inf_logprob

        denom = 1e-6 + abs(val_logprob) + abs(inf_logprob)
        if math.isnan(denom) or denom == 0:
            continue
        term = abs(val_logprob - inf_logprob) / denom / 2.0
        if not math.isnan(term):
            dist += term

    return dist, n_matches


_BAD_LOGPROB_FLOOR = -9990.0


def _token_distance2_core(
    inf_position_logprobs: PositionResult,
    val_position_logprobs: PositionResult,
    skip_inf: bool = False,
    skip_zero: bool = False,
):
    """Shared core for distance2 clean variants.

    Same structure as token_distance2 (Go-aligned: iterates validation tokens,
    fallback from inference side) with optional cleaning:
      skip_inf  — skip pairs where either logprob <= _BAD_LOGPROB_FLOOR (-9999)
      skip_zero — skip pairs where one side is ~0.0 and the other is the max
                  logprob of its position (clamped high-confidence artifact)
    """
    dist = 0.0
    n_matches = 0

    if not inf_position_logprobs.logprobs or not val_position_logprobs.logprobs:
        return 100.0, 0

    sorted_inf_logprobs = sorted(inf_position_logprobs.logprobs.values())

    if len(sorted_inf_logprobs) >= 2:
        min_inf_1 = sorted_inf_logprobs[0]
        min_inf_2 = sorted_inf_logprobs[1]
    else:
        min_inf_1 = sorted_inf_logprobs[0]
        min_inf_2 = min_inf_1 - 100.0

    next_inf_logprob = min_inf_1 - (min_inf_2 - min_inf_1)

    if skip_zero:
        inf_max = max(inf_position_logprobs.logprobs.values())
        val_max = max(val_position_logprobs.logprobs.values())

    for token, val_logprob in val_position_logprobs.logprobs.items():
        if token in inf_position_logprobs.logprobs:
            inf_logprob = inf_position_logprobs.logprobs[token]
            n_matches += 1
        else:
            inf_logprob = next_inf_logprob

        if skip_inf and (inf_logprob <= _BAD_LOGPROB_FLOOR or val_logprob <= _BAD_LOGPROB_FLOOR):
            continue

        if skip_zero:
            inf_is_zero = abs(inf_logprob) < 1e-6
            val_is_zero = abs(val_logprob) < 1e-6
            if inf_is_zero and not val_is_zero and abs(val_logprob - val_max) < 1e-6:
                continue
            if val_is_zero and not inf_is_zero and abs(inf_logprob - inf_max) < 1e-6:
                continue

        denom = 1e-6 + abs(val_logprob) + abs(inf_logprob)
        if math.isnan(denom) or denom == 0:
            continue
        term = abs(val_logprob - inf_logprob) / denom / 2.0
        if not math.isnan(term):
            dist += term

    return dist, n_matches


def _distance2_variant(
    inf_result: Result,
    val_result: Result,
    skip_inf: bool = False,
    skip_zero: bool = False,
):
    if not _check_match(inf_result, val_result):
        return -1, -1

    total_dist = 0
    total_n_matches = 0
    for inf_position, val_position in zip(inf_result.results, val_result.results):
        dist, n_matches = _token_distance2_core(inf_position, val_position, skip_inf=skip_inf, skip_zero=skip_zero)
        total_dist += dist
        total_n_matches += n_matches

    n_logprobs = len(inf_result.results[0].logprobs) if inf_result.results[0].logprobs else 1
    matches_ratio = total_n_matches / (len(inf_result.results) * n_logprobs)
    total_dist = total_dist / (max(100, len(inf_result.results)) * n_logprobs)
    return total_dist, matches_ratio


def distance2_inf_clean(inf_result: Result, val_result: Result):
    """distance2 + skip -9999 pairs."""
    return _distance2_variant(inf_result, val_result, skip_inf=True)


def distance2_zero_clean(inf_result: Result, val_result: Result):
    """distance2 + skip 0.0-vs-max-logprob pairs."""
    return _distance2_variant(inf_result, val_result, skip_zero=True)


def distance2_clean(inf_result: Result, val_result: Result):
    """distance2 + both -9999 and 0.0 cleaning."""
    return _distance2_variant(inf_result, val_result, skip_inf=True, skip_zero=True)


def distance3(inf_result: Result, val_result: Result):
    """Alias for distance2_clean (backward compat)."""
    return distance2_clean(inf_result, val_result)


def similarity2(
    inf_result: Result,
    val_result: Result,
):
    dist, matches_ratio = distance2(inf_result, val_result)
    if dist == -1:
        return -1, -1
    return 1 - dist, matches_ratio


def distance2(
    inf_result: Result,
    val_result: Result,
):
    if not _check_match(inf_result, val_result):
        return -1, -1

    total_dist = 0
    total_n_matches = 0
    for inf_position, val_position in zip(inf_result.results, val_result.results):
        dist, n_matches = token_distance2(inf_position, val_position)
        total_dist += dist
        total_n_matches += n_matches

    n_logprobs = len(inf_result.results[0].logprobs) if inf_result.results[0].logprobs else 1
    matches_ratio = total_n_matches / (len(inf_result.results) * n_logprobs)
    total_dist = total_dist / (max(100, len(inf_result.results)) * n_logprobs)
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
