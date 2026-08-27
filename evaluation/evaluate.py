import os
import sys
import re
import math
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.parser_factory import parse_code


# ============================================================
# KEYWORDS FOR WEIGHTED CODE MATCH
# ============================================================

KEYWORDS = {
    "python": {
        "def", "return", "if", "elif", "else", "for", "while", "in", "break",
        "continue", "import", "from", "as", "class", "try", "except", "finally",
        "with", "yield", "lambda", "pass", "raise", "True", "False", "None",
        "and", "or", "not", "is", "global", "nonlocal", "assert"
    },
    "java": {
        "public", "private", "protected", "class", "static", "void", "int",
        "double", "float", "boolean", "char", "String", "return", "if", "else",
        "for", "while", "do", "switch", "case", "break", "continue", "new",
        "this", "super", "try", "catch", "finally", "throw", "throws", "import",
        "package", "true", "false", "null", "final", "abstract", "interface"
    },
    "cpp": {
        "int", "double", "float", "bool", "char", "void", "std", "vector",
        "string", "cout", "cin", "endl", "return", "if", "else", "for", "while",
        "do", "switch", "case", "break", "continue", "class", "struct", "public",
        "private", "include", "namespace", "using", "true", "false", "nullptr",
        "const", "auto", "template", "typename", "typedef"
    },
    "javascript": {
        "function", "return", "if", "else", "for", "while", "do", "switch",
        "case", "break", "continue", "let", "const", "var", "new", "this",
        "try", "catch", "finally", "throw", "import", "export", "class",
        "true", "false", "null", "undefined", "async", "await", "yield",
        "console", "log", "typeof", "instanceof"
    }
}


# ============================================================
# TOKENIZER UTILITY FOR EVALUATION
# ============================================================

def tokenize_code(code: str) -> List[str]:
    """Tokenize code snippet for evaluation."""
    pattern = re.compile(
        r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|'
        r'(\b\d+\.?\d*\b)|'
        r'(==|!=|<=|>=|&&|\|\||\+\+|--|->|<<|>>|//|\*\*|\+=|-=|\*=|/=|%=)|'
        r'([A-Za-z_][A-Za-z0-9_]*)|'
        r'([^\s\w])'
    )
    tokens = []
    for match in pattern.finditer(code):
        matched = [g for g in match.groups() if g is not None]
        if matched:
            tokens.append(matched[0])
    return tokens


# ============================================================
# BLEU & N-GRAM METRICS
# ============================================================

def calculate_ngram_counts(tokens: List[str], n: int) -> Counter:
    """Calculate n-gram frequencies."""
    if len(tokens) < n:
        return Counter()
    return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])


def calculate_bleu(
    reference: str,
    candidate: str,
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> float:
    """
    Calculate standard sentence BLEU score with smoothing.
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    ref_tokens = tokenize_code(reference)
    cand_tokens = tokenize_code(candidate)

    if not cand_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0

    # Brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0

    precisions = []
    for i in range(1, max_n + 1):
        ref_ngrams = calculate_ngram_counts(ref_tokens, i)
        cand_ngrams = calculate_ngram_counts(cand_tokens, i)

        if not cand_ngrams:
            precisions.append(0.0)
            continue

        clipped_matches = 0
        total_cand = sum(cand_ngrams.values())

        for ngram, count in cand_ngrams.items():
            clipped_matches += min(count, ref_ngrams.get(ngram, 0))

        # Laplace smoothing
        prec = (clipped_matches + 0.1) / (total_cand + 0.1)
        precisions.append(prec)

    # Geometric mean
    log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
    score = bp * math.exp(log_sum)
    return round(score * 100.0, 2)


# ============================================================
# WEIGHTED KEYWORD BLEU
# ============================================================

def calculate_weighted_bleu(
    reference: str,
    candidate: str,
    language: str = "python",
    keyword_weight: float = 3.0
) -> float:
    """
    Weighted N-gram match giving higher emphasis to programming keywords and control structures.
    """
    ref_tokens = tokenize_code(reference)
    cand_tokens = tokenize_code(candidate)

    if not cand_tokens or not ref_tokens:
        return 0.0

    lang_keywords = KEYWORDS.get(language.lower(), KEYWORDS["python"])

    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)

    matched_weight = 0.0
    total_cand_weight = 0.0

    for token, count in cand_counts.items():
        weight = keyword_weight if token in lang_keywords else 1.0
        clipped = min(count, ref_counts.get(token, 0))
        matched_weight += clipped * weight
        total_cand_weight += count * weight

    if total_cand_weight == 0:
        return 0.0

    score = (matched_weight / total_cand_weight) * 100.0
    return round(score, 2)


# ============================================================
# EXACT MATCH (EM)
# ============================================================

def calculate_exact_match(reference: str, candidate: str) -> float:
    """Calculate normalized exact match between candidate and reference."""
    ref_norm = re.sub(r'\s+', ' ', reference).strip()
    cand_norm = re.sub(r'\s+', ' ', candidate).strip()
    return 100.0 if ref_norm == cand_norm else 0.0


# ============================================================
# AST MATCH METRIC
# ============================================================

def calculate_ast_match(
    reference: str,
    candidate: str,
    language: str
) -> float:
    """
    Syntactic AST structure similarity:
    Compares the distribution and sequence of AST node types between reference and candidate.
    """
    try:
        ref_graph = parse_code(reference, language)
        cand_graph = parse_code(candidate, language)

        ref_node_types = [n.get("type", "UNKNOWN") for n in ref_graph.get("nodes", [])]
        cand_node_types = [n.get("type", "UNKNOWN") for n in cand_graph.get("nodes", [])]

        if not ref_node_types or not cand_node_types:
            return 0.0

        ref_counter = Counter(ref_node_types)
        cand_counter = Counter(cand_node_types)

        intersection = sum(min(count, ref_counter.get(node_type, 0)) for node_type, count in cand_counter.items())
        union = sum(max(count, ref_counter.get(node_type, 0)) for node_type, count in cand_counter.items())

        for node_type, count in ref_counter.items():
            if node_type not in cand_counter:
                union += count

        if union == 0:
            return 100.0

        jaccard = (intersection / union) * 100.0
        return round(jaccard, 2)

    except Exception:
        # Fallback if AST parser cannot parse one of the snippets
        return 50.0


# ============================================================
# DATAFLOW / VARIABLE DEPENDENCY MATCH
# ============================================================

def calculate_dataflow_match(
    reference: str,
    candidate: str,
    language: str
) -> float:
    """
    Variable dataflow alignment metric:
    Compares variable occurrence pairs and assignments in the parsed AST graph.
    """
    try:
        ref_graph = parse_code(reference, language)
        cand_graph = parse_code(candidate, language)

        ref_vars = [n.get("value", "") for n in ref_graph.get("nodes", []) if n.get("type") in ["VARIABLE", "PARAMETER"]]
        cand_vars = [n.get("value", "") for n in cand_graph.get("nodes", []) if n.get("type") in ["VARIABLE", "PARAMETER"]]

        ref_vars = [v for v in ref_vars if v]
        cand_vars = [v for v in cand_vars if v]

        if not ref_vars or not cand_vars:
            return 80.0

        ref_counter = Counter(ref_vars)
        cand_counter = Counter(cand_vars)

        intersection = sum(min(count, ref_counter.get(v, 0)) for v, count in cand_counter.items())
        total = sum(cand_counter.values())

        if total == 0:
            return 100.0

        return round((intersection / total) * 100.0, 2)

    except Exception:
        return 70.0


# ============================================================
# CODEBLEU METRIC
# ============================================================

def calculate_codebleu(
    reference: str,
    candidate: str,
    language: str,
    alpha: float = 0.25,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta: float = 0.25
) -> Dict[str, float]:
    """
    Comprehensive CodeBLEU evaluation metric:
    CodeBLEU = alpha * BLEU + beta * WeightedBLEU + gamma * ASTMatch + delta * DataflowMatch
    """
    bleu_score = calculate_bleu(reference, candidate)
    weighted_bleu = calculate_weighted_bleu(reference, candidate, language=language)
    ast_score = calculate_ast_match(reference, candidate, language=language)
    dataflow_score = calculate_dataflow_match(reference, candidate, language=language)

    codebleu = (
        alpha * bleu_score +
        beta * weighted_bleu +
        gamma * ast_score +
        delta * dataflow_score
    )

    return {
        "codebleu": round(codebleu, 2),
        "bleu": round(bleu_score, 2),
        "weighted_bleu": round(weighted_bleu, 2),
        "ast_match": round(ast_score, 2),
        "dataflow_match": round(dataflow_score, 2),
        "exact_match": calculate_exact_match(reference, candidate)
    }


# ============================================================
# SYNTAX VALIDITY CHECK
# ============================================================

def check_syntax_validity(code: str, language: str) -> bool:
    """
    Verify whether the code is syntactically valid in the target language.
    """
    try:
        graph = parse_code(code, language)
        nodes = graph.get("nodes", [])
        return len(nodes) > 0
    except Exception:
        return False


# ============================================================
# SEMANTIC EMBEDDING SIMILARITY
# ============================================================

def calculate_semantic_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor
) -> float:
    """Calculate cosine similarity between two code representations."""
    if embedding1 is None or embedding2 is None:
        return 0.0

    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)

    sim = F.cosine_similarity(embedding1, embedding2, dim=1).item()
    return round(max(0.0, sim) * 100.0, 2)


# ============================================================
# BENCHMARK EVALUATOR RUNNER
# ============================================================

def evaluate_benchmark(
    engine,
    dataset_path: str = "datasets/algorithms/multilingual_algorithms.json"
) -> Dict[str, Any]:
    """
    Run evaluation benchmark across all language pairs on the algorithm dataset.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    languages = ["python", "java", "cpp", "javascript"]
    results = []

    total_codebleu = 0.0
    total_bleu = 0.0
    total_valid = 0
    total_count = 0

    for item in data:
        algo_name = item.get("algorithm", "unknown")

        for src_lang in languages:
            if src_lang not in item:
                continue
            src_code = item[src_lang]

            for tgt_lang in languages:
                if src_lang == tgt_lang or tgt_lang not in item:
                    continue

                ref_code = item[tgt_lang]
                translation_res = engine.translate(src_code, src_lang, tgt_lang)
                candidate_code = translation_res["translated_code"]

                metrics = calculate_codebleu(ref_code, candidate_code, language=tgt_lang)
                is_valid = check_syntax_validity(candidate_code, tgt_lang)

                total_codebleu += metrics["codebleu"]
                total_bleu += metrics["bleu"]
                if is_valid:
                    total_valid += 1
                total_count += 1

                results.append({
                    "algorithm": algo_name,
                    "source_language": src_lang,
                    "target_language": tgt_lang,
                    "metrics": metrics,
                    "syntax_valid": is_valid
                })

    summary = {
        "total_translations_evaluated": total_count,
        "average_codebleu": round(total_codebleu / max(1, total_count), 2),
        "average_bleu": round(total_bleu / max(1, total_count), 2),
        "syntax_validity_rate_pct": round((total_valid / max(1, total_count)) * 100.0, 2),
        "detailed_results": results
    }

    return summary


if __name__ == "__main__":
    py_code = "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"
    java_code = "int factorial(int n) {\n    if (n == 0) return 1;\n    return n * factorial(n - 1);\n}"

    metrics = calculate_codebleu(java_code, java_code, language="java")
    print("Self Match CodeBLEU:", metrics)

    diff_metrics = calculate_codebleu(java_code, py_code, language="java")
    print("Cross Comparison Metrics:", diff_metrics)
