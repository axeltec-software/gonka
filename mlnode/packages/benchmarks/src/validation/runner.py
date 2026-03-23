from typing import List, Optional

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)
from validation.utils import generate_and_validate, generate_inference_snapshot, validate_from_snapshot
from validation.data import (
    ValidationItem,
    InferenceSnapshot,
    ModelInfo,
    RequestParams,
    ExperimentRequest,
)
from tqdm import tqdm
from common.logger import create_logger

logger = create_logger(__name__)


def run_validation(
    prompts: List[str],
    inference_model: ModelInfo,
    validation_model: ModelInfo,
    request_params: RequestParams,
    max_workers: int = 10,
    output_path: Optional[str] = None,
    languages: Optional[List[str]] = None,
) -> List[ValidationItem]:    
    if languages is not None and len(languages) != len(prompts):
        raise ValueError("languages length must match prompts length when provided")

    args = []
    for idx, prompt in enumerate(prompts):
        lang = languages[idx] if languages is not None else None
        args.append(
            ExperimentRequest(
                prompt=prompt,
                language=lang,
                inference_model=inference_model,
                validation_model=validation_model,
                request_params=request_params,
                output_path=output_path,
            )
        )

    results = []
    def submit_one(executor, arg, attempt: int = 1):
        return executor.submit(generate_and_validate, arg)

    max_task_attempts = max(1, request_params.retries_max_attempts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {submit_one(executor, arg): (arg, 1) for arg in args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running validation", leave=False, smoothing=0):
            arg, attempt = futures.pop(future)
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(e)
                if attempt < max_task_attempts:
                    # resubmit with incremented attempt
                    logger.error(f"Task failed (attempt {attempt}/{max_task_attempts}), retrying: {e}")
                    new_future = submit_one(executor, arg, attempt + 1)
                    futures[new_future] = (arg, attempt + 1)
                else:
                    logger.error(f"Task permanently failed after {attempt} attempts: ") #{e}

    return results


def run_inference_only(
    prompts: List[str],
    inference_model: ModelInfo,
    validation_model: ModelInfo,
    request_params: RequestParams,
    max_workers: int = 10,
    output_path: Optional[str] = None,
    languages: Optional[List[str]] = None,
) -> List[InferenceSnapshot]:
    """Run inference for every prompt and save snapshots; skip validation entirely."""
    if languages is not None and len(languages) != len(prompts):
        raise ValueError("languages length must match prompts length when provided")

    args = [
        ExperimentRequest(
            prompt=prompt,
            language=languages[idx] if languages is not None else None,
            inference_model=inference_model,
            validation_model=validation_model,
            request_params=request_params,
            output_path=output_path,
        )
        for idx, prompt in enumerate(prompts)
    ]

    snapshots: List[InferenceSnapshot] = []
    max_task_attempts = max(1, request_params.retries_max_attempts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_inference_snapshot, arg, output_path): (arg, 1) for arg in args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running inference", leave=False, smoothing=0):
            arg, attempt = futures.pop(future)
            try:
                snapshots.append(future.result())
            except Exception as e:
                if attempt < max_task_attempts:
                    logger.error(f"Inference task failed (attempt {attempt}/{max_task_attempts}), retrying: {e}")
                    new_future = executor.submit(generate_inference_snapshot, arg, output_path)
                    futures[new_future] = (arg, attempt + 1)
                else:
                    logger.error(f"Inference task permanently failed after {attempt} attempts: {e}")

    return snapshots


def run_validation_from_snapshots(
    snapshots: List[InferenceSnapshot],
    max_workers: int = 10,
    output_path: Optional[str] = None,
) -> List[ValidationItem]:
    """Run validation over previously saved inference snapshots."""
    results: List[ValidationItem] = []
    max_task_attempts = max(1, snapshots[0].request_params.retries_max_attempts if snapshots else 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_from_snapshot, snap, output_path): (snap, 1) for snap in snapshots}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running validation", leave=False, smoothing=0):
            snap, attempt = futures.pop(future)
            try:
                results.append(future.result())
            except Exception as e:
                if attempt < max_task_attempts:
                    logger.error(f"Validation task failed (attempt {attempt}/{max_task_attempts}), retrying: {e}")
                    new_future = executor.submit(validate_from_snapshot, snap, output_path)
                    futures[new_future] = (snap, attempt + 1)
                else:
                    logger.error(f"Validation task permanently failed after {attempt} attempts: {e}")

    return results
