from parsers.parser_factory import parse_code
import json


code = '''def greet():
    return "Hello"
'''

graph = parse_code(
    code,
    "python"
)

print(json.dumps(
    graph,
    indent=2
))