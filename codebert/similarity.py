import torch
import torch.nn.functional as F

from codebert.encoder import CodeBERTEncoder


def calculate_similarity(embedding1, embedding2):

    similarity = F.cosine_similarity(
        embedding1,
        embedding2
    )

    return similarity.item()


if __name__ == "__main__":

    encoder = CodeBERTEncoder()

    python_code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

    java_code = """
static int factorial(int n) {
    if (n == 0) {
        return 1;
    }
    return n * factorial(n - 1);
}
"""

    cpp_code = """
int factorial(int n) {
    if (n == 0) {
        return 1;
    }
    return n * factorial(n - 1);
}
"""

    javascript_code = """
function factorial(n) {
    if (n === 0) {
        return 1;
    }
    return n * factorial(n - 1);
}
"""

    print("\nGenerating embeddings...")

    python_embedding = encoder.encode(python_code)
    java_embedding = encoder.encode(java_code)
    cpp_embedding = encoder.encode(cpp_code)
    javascript_embedding = encoder.encode(javascript_code)

    print("\nCross-language similarity:")

    print(
        "Python ↔ Java:",
        calculate_similarity(
            python_embedding,
            java_embedding
        )
    )

    print(
        "Python ↔ C++:",
        calculate_similarity(
            python_embedding,
            cpp_embedding
        )
    )

    print(
        "Python ↔ JavaScript:",
        calculate_similarity(
            python_embedding,
            javascript_embedding
        )
    )

    print(
        "Java ↔ C++:",
        calculate_similarity(
            java_embedding,
            cpp_embedding
        )
    )