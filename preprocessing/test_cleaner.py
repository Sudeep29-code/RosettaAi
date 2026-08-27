from datasets import load_from_disk

from preprocessing.cleaner import (
    clean_code,
    is_valid_example
)


dataset = load_from_disk(
    "datasets/raw/codesearchnet/python"
)


valid = 0
invalid = 0


for example in dataset:

    if is_valid_example(example):

        valid += 1

    else:

        invalid += 1


print("Total examples:", len(dataset))
print("Valid examples:", valid)
print("Invalid examples:", invalid)