import re
from typing import List, Dict, Optional, Set


class CodeTokenizer:
    """
    Tokenizer designed for programming languages (Python, Java, C++, JavaScript).
    Handles syntax punctuation, camelCase, snake_case identifiers, literals,
    and language conditioning tokens (<py>, <java>, <cpp>, <js>).
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    LANGUAGE_TOKENS = {
        "python": "<py>",
        "java": "<java>",
        "cpp": "<cpp>",
        "c++": "<cpp>",
        "javascript": "<js>",
        "js": "<js>"
    }

    SPECIAL_TOKENS = [
        PAD_TOKEN,
        SOS_TOKEN,
        EOS_TOKEN,
        UNK_TOKEN,
        "<py>",
        "<java>",
        "<cpp>",
        "<js>",
        "<indent>",
        "<dedent>",
        "<newline>"
    ]

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

        if vocab:
            self.vocab = dict(vocab)
            self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        else:
            self._init_default_vocab()

    def _init_default_vocab(self):
        """Initialize vocabulary with special tokens and common code keywords/symbols."""
        self.vocab = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.vocab[self.SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, code: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize code snippet into tokens, preserving structure, operators, and identifiers.
        """
        tokens = []

        if language:
            lang_key = language.lower().strip()
            if lang_key in self.LANGUAGE_TOKENS:
                tokens.append(self.LANGUAGE_TOKENS[lang_key])

        # Token extraction pattern:
        # 1. String literals ("..." or '...')
        # 2. Numeric literals (int, float)
        # 3. Multi-character operators
        # 4. Words (identifiers, keywords)
        # 5. Newlines
        # 6. Single punctuation / operator characters
        pattern = re.compile(
            r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|'
            r'(\b\d+\.?\d*\b)|'
            r'(==|!=|<=|>=|&&|\|\||\+\+|--|->|<<|>>|//|\*\*|\+=|-=|\*=|/=|%=)|'
            r'([A-Za-z_][A-Za-z0-9_]*)|'
            r'(\n)|'
            r'(\s+)|'
            r'([^\s\w])'
        )

        for match in pattern.finditer(code):
            string_lit, num_lit, multi_op, word, newline, ws, punct = match.groups()

            if string_lit:
                tokens.append(string_lit)
            elif num_lit:
                tokens.append(num_lit)
            elif multi_op:
                tokens.append(multi_op)
            elif word:
                tokens.append(word)
            elif newline:
                tokens.append("<newline>")
            elif punct:
                tokens.append(punct)

        return tokens

    def build_vocab(self, code_samples: List[str], min_freq: int = 1, max_vocab_size: int = 10000):
        """Build vocabulary from a corpus of code samples."""
        self._init_default_vocab()
        frequencies: Dict[str, int] = {}

        for code in code_samples:
            for token in self.tokenize(code):
                if token not in self.vocab:
                    frequencies[token] = frequencies.get(token, 0) + 1

        sorted_tokens = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)

        for token, count in sorted_tokens:
            if count >= min_freq and len(self.vocab) < max_vocab_size:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token

    def encode(
        self,
        code: str,
        language: Optional[str] = None,
        max_length: int = 256,
        add_special_tokens: bool = True
    ) -> List[int]:
        """Convert code text into a list of token IDs."""
        raw_tokens = self.tokenize(code, language=language)

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.sos_id)

        for token in raw_tokens:
            token_ids.append(self.vocab.get(token, self.unk_id))

        if add_special_tokens:
            token_ids.append(self.eos_id)

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            if add_special_tokens:
                token_ids[-1] = self.eos_id

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back into code text with formatting."""
        tokens = []
        for idx in token_ids:
            token = self.inverse_vocab.get(idx, self.UNK_TOKEN)
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                if token == "<newline>":
                    tokens.append("\n")
                continue
            if token == "<newline>":
                tokens.append("\n")
            else:
                tokens.append(token)

        result = []
        for i, token in enumerate(tokens):
            if token == "\n":
                result.append("\n")
            elif token in [";", ",", ")", "]", "}", ":"]:
                if result and result[-1] == " ":
                    result.pop()
                result.append(token)
                if token in [";", ","]:
                    result.append(" ")
            elif token in ["(", "[", "{"]:
                result.append(token)
            else:
                result.append(token)
                result.append(" ")

        return "".join(result).strip()

