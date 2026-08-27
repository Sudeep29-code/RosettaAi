import os
import json
import gzip
from typing import List, Dict, Any


DATASET_PATH = "datasets/algorithms/multilingual_algorithms.json"


def load_dataset(path: str = DATASET_PATH) -> List[Dict[str, Any]]:
    """Load dataset from .json, .jsonl, or .jsonl.gz files."""
    if not os.path.exists(path):
        return []

    if path.endswith(".gz"):
        items = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))
        return items
    elif path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))
        return items
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_humaneval_x_directory(data_dir: str = "datasets/humaneval_x") -> List[Dict[str, Any]]:
    """
    Load and pair humaneval_python.jsonl.gz, humaneval_java.jsonl.gz, 
    humaneval_cpp.jsonl.gz, and humaneval_js.jsonl.gz files into parallel dataset items.
    """
    languages = {
        "python": ["humaneval_python.jsonl.gz", "humaneval_py.jsonl.gz", "python.jsonl"],
        "java": ["humaneval_java.jsonl.gz", "java.jsonl"],
        "cpp": ["humaneval_cpp.jsonl.gz", "cpp.jsonl"],
        "javascript": ["humaneval_js.jsonl.gz", "humaneval_javascript.jsonl.gz", "js.jsonl"]
    }

    lang_data = {}

    for lang, file_candidates in languages.items():
        found = False
        for fname in file_candidates:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                raw_items = load_dataset(fpath)
                lang_data[lang] = {
                    item.get("task_id", f"{lang}/{idx}").split("/")[-1]: item
                    for idx, item in enumerate(raw_items)
                }
                found = True
                print(f"Loaded {len(lang_data[lang])} samples for {lang.upper()} from {fname}")
                break

    # Pair across common task IDs
    if not lang_data:
        return []

    # Use python keys or first available language keys
    first_lang = list(lang_data.keys())[0]
    task_keys = list(lang_data[first_lang].keys())

    parallel_pairs = []
    for key in task_keys:
        pair_item = {"task_id": key}
        for lang in ["python", "java", "cpp", "javascript"]:
            if lang in lang_data and key in lang_data[lang]:
                entry = lang_data[lang][key]
                prompt = entry.get("prompt", "")
                solution = entry.get("canonical_solution", "")
                pair_item[lang] = (prompt + solution).strip()
            else:
                pair_item[lang] = ""
        parallel_pairs.append(pair_item)

    return parallel_pairs


if __name__ == "__main__":
    dataset = load_dataset()
    print("Total algorithms:", len(dataset))
    for item in dataset[:5]:
        print(item.get("algorithm", item.get("task_id", "unnamed")), "-> Python, Java, C++, JS")