from datasets import load_dataset
import os


INPUT_PATH = "datasets/raw/codesearchnet/cpp"


def download_cpp():

    print("Downloading C++ dataset...")

    dataset = load_dataset(
        "malteklaes/cpp-code-code_search_net-style"
    )

    os.makedirs(
        INPUT_PATH,
        exist_ok=True
    )

    # Use train split
    train = dataset["train"]

    # First 100 examples for our prototype
    train = train.select(
        range(min(100, len(train)))
    )

    print("Examples:", len(train))

    train.save_to_disk(
        INPUT_PATH
    )

    print(
        "C++ dataset saved to:",
        INPUT_PATH
    )


if __name__ == "__main__":

    download_cpp()