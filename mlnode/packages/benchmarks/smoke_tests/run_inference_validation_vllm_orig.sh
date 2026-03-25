#!/bin/env bash
set -e
python3 ../scripts/inference.py --type parallel --enforced-tokens-in payload
python3 ../scripts/inference.py --type sequential --mode both --enforced-tokens-in payload
