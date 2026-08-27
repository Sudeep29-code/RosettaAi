from parsers.python_ast import analyze_code as analyze_python
from parsers.java_ast import analyze_java_code
from parsers.cpp_ast import analyze_cpp_code
from parsers.javascript_ast import analyze_javascript_code


PARSERS = {

    "python": analyze_python,

    "java": analyze_java_code,

    "cpp": analyze_cpp_code,

    "c++": analyze_cpp_code,

    "javascript": analyze_javascript_code,

    "js": analyze_javascript_code
}


def parse_code(code, language):

    language = language.lower().strip()

    if language not in PARSERS:

        supported = ", ".join(
            PARSERS.keys()
        )

        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages: {supported}"
        )

    return PARSERS[language](
        code
    )