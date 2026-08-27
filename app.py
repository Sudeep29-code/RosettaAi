import os
from dotenv import load_dotenv

_base_dir = os.path.dirname(__file__)
for _p in [os.path.join(_base_dir, "venv", ".env"), os.path.join(_base_dir, ".env"), os.path.expanduser("~/.env")]:
    if os.path.exists(_p):
        load_dotenv(_p, override=True)
        break

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from translation.rosetta_engine import RosettaEngine
from evaluation.evaluate import (
    calculate_codebleu,
    calculate_bleu,
    calculate_exact_match,
    check_syntax_validity,
    calculate_semantic_similarity,
    evaluate_benchmark
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Rosetta AI - Cross-Lingual Code Translation Engine",
    description=(
        "Advanced Generative AI system for cross-lingual code understanding, "
        "semantic parsing, AST representation, and neural translation across "
        "Python, Java, C++, and JavaScript."
    ),
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://rosettaai-2.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INITIALIZE ENGINE
# ============================================================

print("=" * 60)
print("INITIALIZING ROSETTA AI API")
print("=" * 60)

engine = RosettaEngine()

print("=" * 60)
print("ROSETTA AI API READY")
print("=" * 60)


# ============================================================
# REQUEST & RESPONSE SCHEMAS
# ============================================================

class TranslationRequest(BaseModel):
    code: str = Field(..., description="Source code snippet to translate")
    source_language: str = Field(..., description="Source programming language (python, java, cpp, javascript)")
    target_language: str = Field(..., description="Target programming language (python, java, cpp, javascript)")
    strategy: Optional[str] = Field("hybrid", description="Translation strategy: hybrid, ast, neural, or rules")
    refactor: Optional[bool] = Field(True, description="Whether to apply target language best practices and refactoring")
    use_ai_verifier: Optional[bool] = Field(True, description="Enable Gemini AI translation & polish layer")
    gemini_api_key: Optional[str] = Field(None, description="Optional private Gemini API Key (masked in logs/responses)")
    optimize: Optional[bool] = Field(False, description="Translate and optimize code for optimal execution complexity")


class TranslationResponse(BaseModel):
    source_language: str
    target_language: str
    translated_code: str
    strategy_used: str
    ast_translation: Optional[str] = None
    neural_translation: Optional[str] = None
    rule_translation: Optional[str] = None
    embedding_shape: Optional[List[int]] = None
    predicted_language: Optional[int] = None
    syntax_valid: bool
    ai_verified: Optional[bool] = False
    ai_verifier_status: Optional[str] = "disabled"


class EvaluationRequest(BaseModel):
    reference_code: str = Field(..., description="Ground truth or reference implementation")
    candidate_code: str = Field(..., description="Translated or generated code candidate")
    language: str = Field(..., description="Target language of the code snippets")


class EvaluationResponse(BaseModel):
    language: str
    codebleu: float
    bleu: float
    weighted_bleu: float
    ast_match: float
    dataflow_match: float
    exact_match: float
    syntax_valid: bool


class SemanticRequest(BaseModel):
    code: str = Field(..., description="Code snippet to analyze")
    language: str = Field(..., description="Programming language of the code snippet")


import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================
# ROOT & HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "name": "Rosetta AI",
        "version": "2.0.0",
        "status": "online",
        "supported_languages": ["python", "java", "cpp", "javascript"]
    }


@app.get("/api/info")
def api_info():
    return {
        "name": "Rosetta AI",
        "version": "2.0.0",
        "status": "online",
        "supported_languages": ["python", "java", "cpp", "javascript"],
        "capabilities": [
            "Cross-Lingual Code Translation",
            "AST Graph Analysis & Semantic Normalization",
            "Neural Transformer Seq2Seq Generation",
            "CodeBLEU & Syntactic Evaluation",
            "Target Language Refactoring Patterns"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "engine": "ready",
        "gnn_loaded": engine.semantic_translator.gnn_available,
        "transformer_loaded": engine.neural_generator is not None
    }


@app.get("/languages")
def get_languages():
    return {
        "languages": [
            {
                "name": "Python",
                "id": "python",
                "aliases": ["python", "py"],
                "ast_parser": "Python Built-in AST Parser"
            },
            {
                "name": "Java",
                "id": "java",
                "aliases": ["java"],
                "ast_parser": "Javalang Method & Class Parser"
            },
            {
                "name": "C++",
                "id": "cpp",
                "aliases": ["cpp", "c++"],
                "ast_parser": "Tree-Sitter C++ Grammar Parser"
            },
            {
                "name": "JavaScript",
                "id": "javascript",
                "aliases": ["javascript", "js"],
                "ast_parser": "Tree-Sitter JavaScript Grammar Parser"
            }
        ]
    }


# ============================================================
# TRANSLATION ENDPOINT
# ============================================================

@app.post("/translate", response_model=TranslationResponse)
def translate_code(request: TranslationRequest):
    if not request.code.strip():
        raise HTTPException(
            status_code=400,
            detail="Code cannot be empty."
        )

    source = request.source_language.lower().strip()
    target = request.target_language.lower().strip()
    supported = {"python", "java", "javascript", "js", "cpp", "c++"}

    if source not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source language: {source}."
        )

    if target not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language: {target}."
        )

    try:
        result = engine.translate(
            code=request.code,
            source_language=source,
            target_language=target,
            strategy=request.strategy or "hybrid",
            refactor=request.refactor if request.refactor is not None else True,
            use_ai_verifier=request.use_ai_verifier or False,
            gemini_api_key=request.gemini_api_key,
            optimize=request.optimize or False
        )

        # Drop tensor for JSON serialization
        result.pop("embedding", None)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


@app.post("/optimize", response_model=TranslationResponse)
def optimize_code(request: TranslationRequest):
    request.optimize = True
    request.use_ai_verifier = True
    return translate_code(request)



# ============================================================
# EVALUATION ENDPOINT
# ============================================================

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_code(request: EvaluationRequest):
    lang = request.language.lower().strip()

    try:
        metrics = calculate_codebleu(
            reference=request.reference_code,
            candidate=request.candidate_code,
            language=lang
        )
        is_valid = check_syntax_validity(request.candidate_code, lang)

        return {
            "language": lang,
            "codebleu": metrics["codebleu"],
            "bleu": metrics["bleu"],
            "weighted_bleu": metrics["weighted_bleu"],
            "ast_match": metrics["ast_match"],
            "dataflow_match": metrics["dataflow_match"],
            "exact_match": metrics["exact_match"],
            "syntax_valid": is_valid
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


# ============================================================
# SEMANTIC REPRESENTATION ENDPOINT
# ============================================================

@app.post("/semantic")
def get_semantic_representation(request: SemanticRequest):
    lang = request.language.lower().strip()

    try:
        graph = engine.semantic_translator.build_graph(request.code, lang)
        output, embedding = engine.semantic_translator.get_embedding(request.code, lang)

        return {
            "language": lang,
            "num_ast_nodes": len(graph.get("nodes", [])),
            "num_ast_edges": len(graph.get("edges", [])),
            "embedding_dimension": list(embedding.shape),
            "embedding_sample": embedding[0, :8].tolist() if embedding is not None else []
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic analysis failed: {str(e)}"
        )


# ============================================================
# BENCHMARK ENDPOINT
# ============================================================

@app.get("/benchmark")
def run_benchmark():
    try:
        summary = evaluate_benchmark(engine)
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark execution failed: {str(e)}"
        )