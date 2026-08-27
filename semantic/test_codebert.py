from semantic.codebert import CodeBERTEncoder


code = """
def factorial(n):
    if n == 0:
        return 1

    return n * factorial(n - 1)
"""


encoder = CodeBERTEncoder()

embedding = encoder.encode(
    code
)

print(
    "Embedding shape:",
    embedding.shape
)