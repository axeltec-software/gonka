#!/bin/env bash
set -e
python3 ../scripts/inference.py --type parallel --enforced-tokens-in vllm
python3 ../scripts/inference.py --type sequential --mode both --enforced-tokens-in vllm
