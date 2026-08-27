"""
Large-Scale Dataset Downloader and Preprocessor for Rosetta AI.

Supported Sources:
1. MultiPL-E: Parallel HumanEval solutions across Python, Java, C++, JS.
2. CodeXGLUE (Microsoft): 100k+ parallel Java-Python functions.
3. CodeSearchNet: 2M+ functions in Python, Java, JS, C++.
4. Custom LeetCode / Algorithmic repos.
"""

import os
import json
import argparse
from typing import List, Dict, Any


def download_multiple_e(output_path: str = "datasets/parallel/multiple_e_parallel.json", limit: int = 164):
    """
    Downloads parallel HumanEval benchmarks in Python, Java, C++, and JavaScript from MultiPL-E.
    """
    print("=" * 60)
    print("Fetching MultiPL-E Benchmark Dataset...")
    print("=" * 60)
    try:
        from datasets import load_dataset
        
        py_ds = load_dataset("nuprl/MultiPL-E", "humaneval-py", split="test")
        java_ds = load_dataset("nuprl/MultiPL-E", "humaneval-java", split="test")
        cpp_ds = load_dataset("nuprl/MultiPL-E", "humaneval-cpp", split="test")
        js_ds = load_dataset("nuprl/MultiPL-E", "humaneval-js", split="test")

        parallel_items = []
        num_items = min(limit, len(py_ds), len(java_ds), len(cpp_ds), len(js_ds))

        for i in range(num_items):
            py_code = py_ds[i].get("prompt", "") + py_ds[i].get("canonical_solution", "")
            java_code = java_ds[i].get("prompt", "") + java_ds[i].get("canonical_solution", "")
            cpp_code = cpp_ds[i].get("prompt", "") + cpp_ds[i].get("canonical_solution", "")
            js_code = js_ds[i].get("prompt", "") + js_ds[i].get("canonical_solution", "")

            parallel_items.append({
                "id": f"multiple_e_{i}",
                "name": py_ds[i].get("name", f"problem_{i}"),
                "python": py_code.strip(),
                "java": java_code.strip(),
                "cpp": cpp_code.strip(),
                "javascript": js_code.strip()
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parallel_items, f, indent=4)

        print(f"Successfully saved {len(parallel_items)} MultiPL-E parallel pairs to {output_path}")
        return parallel_items
    except Exception as e:
        print(f"Notice: Could not download MultiPL-E: {e}")
        return []


def download_codesearchnet_subset(output_dir: str = "datasets/raw/codesearchnet", samples_per_lang: int = 500):
    """
    Downloads subset of CodeSearchNet for Python, Java, JavaScript, and C++.
    """
    print("=" * 60)
    print(f"Fetching CodeSearchNet subset ({samples_per_lang} samples per language)...")
    print("=" * 60)
    from datasets import load_dataset

    languages = {
        "python": "google-research-datasets/codesearchnet",
        "java": "google-research-datasets/codesearchnet",
        "javascript": "google-research-datasets/codesearchnet",
        "cpp": "malteklaes/cpp-code-code_search_net-style"
    }

    for lang, hf_id in languages.items():
        try:
            print(f"Downloading {lang.upper()} from {hf_id}...")
            save_path = os.path.join(output_dir, lang)
            os.makedirs(save_path, exist_ok=True)

            if lang == "cpp":
                ds = load_dataset(hf_id, split="train")
            else:
                ds = load_dataset(hf_id, lang, split="train")

            subset = ds.select(range(min(samples_per_lang, len(ds))))
            subset.save_to_disk(save_path)
            print(f"Saved {len(subset)} {lang} samples to {save_path}")
        except Exception as e:
            print(f"Warning downloading {lang}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download large-scale datasets for Rosetta AI")
    parser.add_argument("--source", type=str, default="all", choices=["all", "multipl_e", "codesearchnet"])
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples per language")
    args = parser.parse_args()

    if args.source in ["all", "multipl_e"]:
        download_multiple_e(limit=args.samples)

    if args.source in ["all", "codesearchnet"]:
        download_codesearchnet_subset(samples_per_lang=args.samples)


if __name__ == "__main__":
    main()

