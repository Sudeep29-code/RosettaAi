from parsers.parser_factory import parse_code
import json

code = """def check_age(age):
    if age >= 18:
        return True
    else:
        return False
"""

graph = parse_code(
    code,
    "python"
)

print(
    json.dumps(
        graph,
        indent=2
    )
)