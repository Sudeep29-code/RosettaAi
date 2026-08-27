import pytest

from translation.rosetta_engine import RosettaEngine


@pytest.fixture(scope="module")
def rosetta():
    return RosettaEngine()


def translate(rosetta, code):
    result = rosetta.translate(
        code,
        source_language="python",
        target_language="java"
    )

    assert result is not None
    assert isinstance(result, dict)

    return result["translated_code"]


def test_addition(rosetta):
    code = """
def add(a, b):
    result = a + b
    return result
"""

    result = translate(rosetta, code)

    assert "add" in result
    assert "return result;" in result


def test_condition(rosetta):
    code = """
def check(x):
    if x > 10:
        return True
    return False
"""

    result = translate(rosetta, code)

    assert "if (x > 10)" in result
    assert "return true;" in result
    assert "return false;" in result


def test_integer(rosetta):
    code = """
def get_number():
    return 10
"""

    result = translate(rosetta, code)

    assert "return 10;" in result
    assert "int get_number" in result


def test_string(rosetta):
    code = """
def greet():
    return "Hello"
"""

    result = translate(rosetta, code)

    assert 'return "Hello";' in result
    assert "String greet" in result


def test_subtraction(rosetta):
    code = """
def sub(a, b):
    result = a - b
    return result
"""

    result = translate(rosetta, code)

    assert "result = a - b;" in result


def test_multiplication(rosetta):
    code = """
def multiply(a, b):
    result = a * b
    return result
"""

    result = translate(rosetta, code)

    assert "result = a * b;" in result


def test_if_else(rosetta):
    code = """
def check_age(age):
    if age >= 18:
        return "Adult"
    else:
        return "Minor"
"""

    result = translate(rosetta, code)

    assert "if (age >= 18)" in result
    assert 'return "Adult";' in result
    assert 'return "Minor";' in result


def test_boolean_variable(rosetta):
    code = """
def is_valid():
    active = True
    return active
"""

    result = translate(rosetta, code)

    assert "boolean active = true;" in result
    assert "return active;" in result


def test_string_variable(rosetta):
    code = """
def message():
    text = "Hello World"
    return text
"""

    result = translate(rosetta, code)

    assert 'String text = "Hello World";' in result
    assert "return text;" in result


def test_while_loop(rosetta):
    code = """
def count():
    x = 0

    while x < 5:
        print(x)
        x = x + 1

    return x
"""

    result = translate(rosetta, code)

    assert "while (x < 5)" in result
    assert "System.out.println(x);" in result


def test_for_loop(rosetta):
    code = """
def calculate(numbers):
    total = 0

    for x in numbers:
        total = total + x

    return total
"""

    result = translate(rosetta, code)

    assert "for" in result
    assert "total = total + x;" in result
    assert "return total;" in result