def clean_code(code):

    if code is None:
        return None

    code = str(code).strip()

    if len(code) == 0:
        return None

    return code


def is_valid_example(example):

    code = example.get("func_code_string")

    return clean_code(code) is not None