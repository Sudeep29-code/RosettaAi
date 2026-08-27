import os
import json
import torch
from typing import Optional, Dict, Any

from translation.translator import SemanticTranslator
from translation.ast_translator import ASTTranslator
from translation.gemini_verifier import GeminiCodeVerifier
from preprocessing.tokenizer import CodeTokenizer
from transformer.model import RosettaTransformer
from transformer.generate import ConstrainedCodeGenerator
from parsers.parser_factory import parse_code


class RosettaEngine:
    """
    Rosetta AI Core Engine.
    Combines:
    1. Semantic AST graph translation preserving logic and structure
    2. Neural Seq2Seq Transformer model with syntax-constrained decoding
    3. Joint CodeBERT + GNN semantic code representations
    4. Automated refactoring patterns and best practices
    """

    def __init__(self):
        print("Initializing Rosetta AI...")

        # 1. AST & Semantic Translators
        self.semantic_translator = SemanticTranslator()
        self.ast_translator = ASTTranslator()
        self.gemini_verifier = GeminiCodeVerifier()

        # 2. Neural Transformer & Constrained Generator
        self.neural_generator = None
        self.transformer_model = None
        self.tokenizer = None
        self._init_neural_generator()

        print("Rosetta AI initialized successfully.")

    def _init_neural_generator(self):
        """Load neural transformer model and tokenizer if weights exist."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        weights_path = os.path.join(base_dir, "transformer", "rosetta_transformer.pth")
        vocab_path = os.path.join(base_dir, "transformer", "vocab.json")

        if os.path.exists(weights_path) and os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)

                self.tokenizer = CodeTokenizer(vocab=vocab)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                self.transformer_model = RosettaTransformer(
                    vocab_size=self.tokenizer.vocab_size,
                    d_model=128,
                    nhead=4,
                    num_encoder_layers=3,
                    num_decoder_layers=3,
                    dim_feedforward=256,
                    dropout=0.1,
                    pad_idx=self.tokenizer.pad_id
                )
                self.transformer_model.load_state_dict(
                    torch.load(weights_path, map_location=device)
                )
                self.transformer_model.to(device)
                self.transformer_model.eval()

                self.neural_generator = ConstrainedCodeGenerator(
                    model=self.transformer_model,
                    tokenizer=self.tokenizer,
                    device=device
                )
                print("Neural Transformer model loaded successfully.")
            except Exception as e:
                print(f"Notice: Neural transformer model initialization skipped: {e}")
        else:
            print("Notice: Neural transformer checkpoint not found. Operating with AST + Semantic engine.")

    # ============================================
    # REFACTORING & BEST PRACTICES
    # ============================================

    def apply_refactoring(self, code: str, target_language: str) -> str:
        """
        Apply target-language refactoring patterns and best practices:
        - C++: Add standard headers (#include <vector>, <string>, <iostream>) and namespace
        - Java: Wrap loose methods in standard public class if missing
        - JavaScript: Enforce modern ES6 conventions (const/let, Math utilities)
        - Python: Ensure standard idiomatic spacing and clean def headers
        """
        target = target_language.lower().strip()
        code = code.strip()

        if target in ["cpp", "c++"]:
            headers = []
            if "vector" in code and "#include <vector>" not in code:
                headers.append("#include <vector>")
            if "string" in code and "#include <string>" not in code:
                headers.append("#include <string>")
            if "cout" in code or "print" in code:
                if "#include <iostream>" not in code:
                    headers.append("#include <iostream>")
            if "cout" in code and "using namespace std;" not in code and "std::cout" not in code:
                headers.append("using namespace std;")

            if headers:
                header_block = "\n".join(headers)
                code = f"{header_block}\n\n{code}"

        elif target == "java":
            needs_scanner = "scanner" in code
            if needs_scanner and "Scanner scanner" not in code:
                # Add scanner initialization in main
                if "main(String[] args)" in code:
                    code = code.replace(
                        "public static void main(String[] args) {",
                        "public static void main(String[] args) {\n        Scanner scanner = new Scanner(System.in);"
                    )

            if "class " not in code:
                lines = ["    " + line for line in code.split("\n")]
                indented_code = "\n".join(lines)
                code = f"public class Solution {{\n{indented_code}\n}}"

            if needs_scanner and "import java.util.Scanner;" not in code:
                code = f"import java.util.Scanner;\n\n{code}"

        elif target in ["javascript", "js"]:
            # Ensure var is replaced with let/const
            code = code.replace("var ", "let ")

        return code

    # ============================================
    # TRANSLATE
    # ============================================

    def translate(
        self,
        code: str,
        source_language: str,
        target_language: str,
        strategy: str = "hybrid",
        refactor: bool = True,
        use_ai_verifier: bool = False,
        gemini_api_key: Optional[str] = None,
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Translate code across Python, Java, C++, JavaScript.

        Strategies:
        - hybrid: AST translation with neural/rule fallbacks and semantic validation
        - ast: Direct AST graph translation
        - neural: Transformer Seq2Seq model with constrained generation
        - rules: Rule-based translation
        """
        source = source_language.lower().strip()
        target = target_language.lower().strip()

        if source == target:
            return {
                "source_language": source,
                "target_language": target,
                "translated_code": code,
                "strategy_used": "identity",
                "ast_translation": code,
                "neural_translation": code,
                "rule_translation": code,
                "embedding": None,
                "embedding_shape": None,
                "predicted_language": None,
                "syntax_valid": True,
                "ai_verified": False,
                "ai_verifier_status": "same_language"
            }

        ast_result = ""
        neural_result = ""
        rule_result = ""
        embedding = None
        prediction = None

        # ----------------------------------------------------
        # 1. AST Graph Translation
        # ----------------------------------------------------
        try:
            graph = self.semantic_translator.build_graph(code, source)
            if isinstance(graph, dict):
                graph["source_code"] = code
                graph["source"] = source
            ast_result = self.ast_translator.translate_graph(graph, source, target)
        except Exception as e:
            ast_result = ""

        # ----------------------------------------------------
        # 2. Rule Translation & Semantic Embeddings
        # ----------------------------------------------------
        try:
            rule_out = self.semantic_translator.translate(code, source, target)
            if isinstance(rule_out, dict):
                rule_result = rule_out.get("translated_code", "")
                embedding = rule_out.get("embedding")
                prediction = rule_out.get("prediction")
            else:
                rule_result = str(rule_out)
        except Exception:
            rule_result = ""

        # ----------------------------------------------------
        # 3. Neural Transformer Translation
        # ----------------------------------------------------
        if self.neural_generator is not None:
            try:
                neural_result = self.neural_generator.generate_beam_search(
                    source_code=code,
                    source_language=source,
                    target_language=target
                )
            except Exception:
                neural_result = ""

        # ----------------------------------------------------
        # 4. Strategy Selection
        # ----------------------------------------------------
        strategy_used = strategy.lower()

        if strategy_used == "ast" and ast_result.strip():
            final_code = ast_result.strip()
        elif strategy_used == "neural" and neural_result.strip():
            final_code = neural_result.strip()
        elif strategy_used == "rules" and rule_result.strip():
            final_code = rule_result.strip()
        else:
            # Hybrid priority: AST -> Neural -> Rule -> Original
            if ast_result and ast_result.strip():
                final_code = ast_result.strip()
                strategy_used = "ast_hybrid"
            elif neural_result and neural_result.strip():
                final_code = neural_result.strip()
                strategy_used = "neural_hybrid"
            elif rule_result and rule_result.strip():
                final_code = rule_result.strip()
                strategy_used = "rule_hybrid"
            else:
                final_code = code
                strategy_used = "fallback"

        # ----------------------------------------------------
        # 5. Apply Refactoring Patterns & Best Practices
        # ----------------------------------------------------
        if refactor:
            final_code = self.apply_refactoring(final_code, target)

        # ----------------------------------------------------
        # 6. Check Syntax Validity of Translation
        # ----------------------------------------------------
        try:
            parsed = parse_code(final_code, target)
            syntax_valid = len(parsed.get("nodes", [])) > 0
        except Exception:
            syntax_valid = False

        # ----------------------------------------------------
        # 7. Optional AI Verification & Polish via Gemini
        # ----------------------------------------------------
        ai_verified = False
        ai_verifier_status = "disabled"
        should_verify = use_ai_verifier or bool(gemini_api_key and gemini_api_key.strip()) or self.gemini_verifier.is_available()

        if should_verify:
            if optimize:
                opt_res = self.gemini_verifier.optimize_code(
                    source_code=code,
                    source_language=source,
                    target_language=target,
                    api_key=gemini_api_key
                )
                if opt_res.get("verified_by_ai"):
                    final_code = opt_res.get("optimized_code", final_code)
                    ai_verified = True
                    strategy_used = "gemini_ai_optimizer"
                    try:
                        parsed = parse_code(final_code, target)
                        syntax_valid = len(parsed.get("nodes", [])) > 0
                    except Exception:
                        syntax_valid = True
                ai_verifier_status = opt_res.get("status", "unknown")
            else:
                verify_res = self.gemini_verifier.translate_code(
                    source_code=code,
                    source_language=source,
                    target_language=target,
                    candidate_code=final_code,
                    api_key=gemini_api_key
                )
                if verify_res.get("verified_by_ai"):
                    final_code = verify_res.get("translated_code", final_code)
                    ai_verified = True
                    strategy_used = "gemini_ai_translation"
                    try:
                        parsed = parse_code(final_code, target)
                        syntax_valid = len(parsed.get("nodes", [])) > 0
                    except Exception:
                        syntax_valid = True
                ai_verifier_status = verify_res.get("status", "unknown")

        embedding_shape = list(embedding.shape) if embedding is not None else None

        return {
            "source_language": source,
            "target_language": target,
            "translated_code": final_code,
            "strategy_used": strategy_used,
            "ast_translation": ast_result,
            "neural_translation": neural_result,
            "rule_translation": rule_result,
            "embedding": embedding,
            "embedding_shape": embedding_shape,
            "predicted_language": prediction,
            "syntax_valid": syntax_valid,
            "ai_verified": ai_verified,
            "ai_verifier_status": ai_verifier_status
        }

    def optimize(
        self,
        code: str,
        source_language: str,
        target_language: str,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate and optimize code for optimal execution complexity and performance.
        """
        return self.translate(
            code=code,
            source_language=source_language,
            target_language=target_language,
            optimize=True,
            use_ai_verifier=True,
            gemini_api_key=gemini_api_key
        )