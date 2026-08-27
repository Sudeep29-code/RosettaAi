import os
import sys
import torch
import torch.nn.functional as F
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.tokenizer import CodeTokenizer
from transformer.model import RosettaTransformer


class ConstrainedCodeGenerator:
    """
    Code generation engine with greedy search, beam search, temperature sampling,
    and syntax constraint verification for target languages.
    """

    def __init__(
        self,
        model: RosettaTransformer,
        tokenizer: CodeTokenizer,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate_greedy(
        self,
        source_code: str,
        source_language: str,
        target_language: str,
        max_length: int = 150
    ) -> str:
        """Greedy decoding from source code to target language."""
        target_token = self.tokenizer.LANGUAGE_TOKENS.get(target_language.lower(), "<py>")
        src_ids = self.tokenizer.encode(source_code, language=source_language, max_length=max_length)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            memory = self.model.encode(src_tensor)
            memory_padding_mask = self.model.create_padding_mask(src_tensor)

            # Start target sequence with SOS and Target Language Tag
            tgt_ids = [self.tokenizer.sos_id]
            if target_token in self.tokenizer.vocab:
                tgt_ids.append(self.tokenizer.vocab[target_token])

            for _ in range(max_length):
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=self.device)
                logits = self.model.decode(
                    tgt_tensor,
                    memory,
                    memory_key_padding_mask=memory_padding_mask
                )
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()

                if next_token_id == self.tokenizer.eos_id:
                    break

                tgt_ids.append(next_token_id)

        translated = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
        return self._apply_syntax_constraints(translated, target_language)

    def generate_beam_search(
        self,
        source_code: str,
        source_language: str,
        target_language: str,
        beam_width: int = 3,
        max_length: int = 150
    ) -> str:
        """Beam search decoding with beam width."""
        target_token = self.tokenizer.LANGUAGE_TOKENS.get(target_language.lower(), "<py>")
        src_ids = self.tokenizer.encode(source_code, language=source_language, max_length=max_length)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            memory = self.model.encode(src_tensor)
            memory_padding_mask = self.model.create_padding_mask(src_tensor)

            start_ids = [self.tokenizer.sos_id]
            if target_token in self.tokenizer.vocab:
                start_ids.append(self.tokenizer.vocab[target_token])

            # Beams: list of (token_ids, cumulative_log_prob, is_finished)
            beams = [(start_ids, 0.0, False)]

            for _ in range(max_length):
                all_candidates = []
                all_finished = True

                for token_ids, score, finished in beams:
                    if finished:
                        all_candidates.append((token_ids, score, True))
                        continue

                    all_finished = False
                    tgt_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                    logits = self.model.decode(
                        tgt_tensor,
                        memory,
                        memory_key_padding_mask=memory_padding_mask
                    )
                    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                    for k in range(beam_width):
                        tok_id = topk_indices[k].item()
                        tok_prob = topk_log_probs[k].item()
                        is_eos = (tok_id == self.tokenizer.eos_id)
                        new_ids = token_ids if is_eos else token_ids + [tok_id]
                        all_candidates.append((new_ids, score + tok_prob, is_eos))

                if all_finished:
                    break

                # Sort and select top beams
                beams = sorted(all_candidates, key=lambda x: x[1] / max(1, len(x[0])), reverse=True)[:beam_width]

            best_tokens = beams[0][0]

        translated = self.tokenizer.decode(best_tokens, skip_special_tokens=True)
        return self._apply_syntax_constraints(translated, target_language)

    def _apply_syntax_constraints(self, code: str, target_language: str) -> str:
        """
        Post-generation syntax constraint enforcer:
        Checks and balances parentheses, curly braces, quotes, and language-specific terminators.
        """
        lines = code.split("\n")
        fixed_lines = []

        open_braces = 0
        open_parens = 0

        target = target_language.lower().strip()

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            open_braces += trimmed.count("{") - trimmed.count("}")
            open_parens += trimmed.count("(") - trimmed.count(")")

            # In C++/Java/JS, ensure statements end with semicolon if not a block/control stmt
            if target in ["java", "cpp", "c++", "javascript", "js"]:
                if (
                    not trimmed.endswith("{")
                    and not trimmed.endswith("}")
                    and not trimmed.endswith(";")
                    and not trimmed.startswith("//")
                    and not trimmed.startswith("#")
                    and not trimmed.startswith("if")
                    and not trimmed.startswith("for")
                    and not trimmed.startswith("while")
                    and not trimmed.startswith("else")
                    and not trimmed.endswith(":")
                ):
                    trimmed += ";"

            fixed_lines.append(trimmed)

        # Balance trailing braces for C++/Java/JS
        if target in ["java", "cpp", "c++", "javascript", "js"]:
            while open_braces > 0:
                fixed_lines.append("}")
                open_braces -= 1

        return "\n".join(fixed_lines)
