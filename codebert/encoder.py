import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "microsoft/codebert-base"


class CodeBERTEncoder:

    def __init__(self):

        print("Loading CodeBERT...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.model.eval()

        print("CodeBERT loaded successfully.")

    def encode(self, code):

        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = self.model(**inputs)

        # Mean pooling
        embedding = outputs.last_hidden_state.mean(
            dim=1
        )

        return embedding


if __name__ == "__main__":

    encoder = CodeBERTEncoder()

    code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

    embedding = encoder.encode(code)

    print("\nEmbedding shape:")
    print(embedding.shape)