import pytest
import torch
from fastapi.testclient import TestClient

from app import app, engine
from parsers.parser_factory import parse_code
from semantic.normalizer import ASTNormalizer
from semantic.codebert import CodeBERTEncoder
from gnn.model import CodeGNN
from preprocessing.tokenizer import CodeTokenizer
from transformer.model import RosettaTransformer
from transformer.generate import ConstrainedCodeGenerator
from evaluation.evaluate import (
    calculate_codebleu,
    calculate_bleu,
    calculate_exact_match,
    check_syntax_validity,
    calculate_semantic_similarity
)


# ============================================================
# 1. PARSER TESTS ACROSS ALL 4 LANGUAGES
# ============================================================

def test_python_parser():
    code = "def add(a, b):\n    return a + b"
    graph = parse_code(code, "python")
    assert "nodes" in graph
    assert "edges" in graph
    assert any(n.get("type") == "FUNCTION" for n in graph["nodes"])


def test_java_parser():
    code = "public static int add(int a, int b) {\n    return a + b;\n}"
    graph = parse_code(code, "java")
    assert "nodes" in graph
    assert "edges" in graph
    assert any(n.get("type") == "FUNCTION" for n in graph["nodes"])


def test_cpp_parser():
    code = "int add(int a, int b) {\n    return a + b;\n}"
    graph = parse_code(code, "cpp")
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) > 0


def test_javascript_parser():
    code = "function add(a, b) {\n    return a + b;\n}"
    graph = parse_code(code, "javascript")
    assert "nodes" in graph
    assert "edges" in graph
    assert any(n.get("type") == "FUNCTION" for n in graph["nodes"])


# ============================================================
# 2. SEMANTIC NORMALIZER TESTS
# ============================================================

def test_ast_normalizer():
    normalizer = ASTNormalizer()
    py_graph = parse_code("def mul(x, y):\n    return x * y", "python")
    norm_graph = normalizer.normalize(py_graph, "python")
    assert "nodes" in norm_graph
    assert len(norm_graph["nodes"]) > 0


# ============================================================
# 3. CODE TOKENIZER TESTS
# ============================================================

def test_code_tokenizer():
    tokenizer = CodeTokenizer()
    code = "def factorial(n):\n    if n == 0:\n        return 1"
    tokens = tokenizer.tokenize(code, language="python")
    assert "<py>" in tokens
    assert "factorial" in tokens
    assert "==" in tokens

    tokenizer.build_vocab([code])
    encoded = tokenizer.encode(code, language="python")
    assert len(encoded) > 0
    decoded = tokenizer.decode(encoded)
    assert "factorial" in decoded


# ============================================================
# 4. NEURAL TRANSFORMER MODEL TESTS
# ============================================================

def test_transformer_model_forward():
    tokenizer = CodeTokenizer()
    model = RosettaTransformer(
        vocab_size=max(50, tokenizer.vocab_size),
        d_model=64,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128
    )
    src = torch.randint(0, 30, (2, 10))
    tgt = torch.randint(0, 30, (2, 8))
    logits = model(src, tgt)
    assert logits.shape == (2, 8, model.vocab_size)


# ============================================================
# 5. CROSS-LINGUAL TRANSLATION (ALL 4 LANGUAGES)
# ============================================================

def test_python_to_java_translation():
    code = "def max_val(a, b):\n    if a > b:\n        return a\n    return b"
    result = engine.translate(code, "python", "java")
    assert result["translated_code"] is not None
    assert "max_val" in result["translated_code"] or "if" in result["translated_code"]


def test_python_to_cpp_translation():
    code = "def add(a, b):\n    return a + b"
    result = engine.translate(code, "python", "cpp")
    assert result["translated_code"] is not None
    assert "add" in result["translated_code"]


def test_python_to_javascript_translation():
    code = "def check(x):\n    if x > 0:\n        return True\n    return False"
    result = engine.translate(code, "python", "javascript")
    assert result["translated_code"] is not None
    assert "function" in result["translated_code"]


def test_java_to_python_translation():
    code = "public static int add(int a, int b) {\n    return a + b;\n}"
    result = engine.translate(code, "java", "python")
    assert result["translated_code"] is not None


def test_javascript_to_python_translation():
    code = "function multiply(a, b) {\n    return a * b;\n}"
    result = engine.translate(code, "javascript", "python")
    assert result["translated_code"] is not None


# ============================================================
# 6. REFACTORING & BEST PRACTICES TESTS
# ============================================================

def test_refactoring_cpp_headers():
    raw_cpp = "int sum(vector<int> arr) {\n    return 0;\n}"
    refactored = engine.apply_refactoring(raw_cpp, "cpp")
    assert "#include <vector>" in refactored


def test_refactoring_java_class():
    raw_java = "public static int square(int x) {\n    return x * x;\n}"
    refactored = engine.apply_refactoring(raw_java, "java")
    assert "public class Solution" in refactored


# ============================================================
# 7. CODEBLEU & EVALUATION METRICS TESTS
# ============================================================

def test_evaluation_metrics():
    ref = "int add(int a, int b) {\n    return a + b;\n}"
    cand = "int add(int a, int b) {\n    return a + b;\n}"
    metrics = calculate_codebleu(ref, cand, language="java")
    assert metrics["exact_match"] == 100.0
    assert metrics["codebleu"] > 90.0

    valid = check_syntax_validity(cand, "java")
    assert valid is True


# ============================================================
# 8. FASTAPI ENDPOINTS TESTS
# ============================================================

@pytest.fixture
def client():
    return TestClient(app)


def test_api_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Rosetta" in response.text


def test_api_info(client):
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_languages(client):
    response = client.get("/languages")
    assert response.status_code == 200
    data = response.json()
    assert len(data["languages"]) == 4


def test_api_translate(client):
    payload = {
        "code": "def double_num(x):\n    return x * 2",
        "source_language": "python",
        "target_language": "java",
        "strategy": "hybrid",
        "refactor": True
    }
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "translated_code" in data
    assert len(data["translated_code"]) > 0


def test_api_evaluate(client):
    payload = {
        "reference_code": "def greet():\n    return 'hello'",
        "candidate_code": "def greet():\n    return 'hello'",
        "language": "python"
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "codebleu" in data
    assert "syntax_valid" in data


def test_api_semantic(client):
    payload = {
        "code": "def compute(a, b):\n    return a + b",
        "language": "python"
    }
    response = client.post("/semantic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "num_ast_nodes" in data
