import json
import sys

INPUT = "mlnode/packages/benchmarks/data/experiments/final_version/inference_fraud_fp8_hack/inference_results.jsonl"
OUTPUT = "mlnode/packages/benchmarks/data/experiments/final_version/inference_fraud_fp8_hack/prompts.txt"


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line)
            prompt = record["prompt"].replace("\n", r"\n")
            inference_text = record["inference_result"]["text"].replace("\n", r"\n")
            fout.write(f"{prompt}{inference_text}\n")

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
