import random
import os
from tokenizer import *

def sample_random_segment(file_path, sample_size_bytes, seed=None):
    """
    Sample a random segment from a text file.
    """
    total_bytes = os.path.getsize(file_path)

    if sample_size_bytes >= total_bytes:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if seed is not None:
        random.seed(seed)

    start_pos = random.randint(0, total_bytes - sample_size_bytes)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(start_pos)
        segment = f.read(sample_size_bytes)  # Approximate byte count in text mode

    return segment


def generate_multiple_files(file_path, list_of_size_mb, seed=None):
    """
    Generate multiple sample files given a list of target sizes
    """
    file_dir = os.path.dirname(file_path)
    temp_dir = os.path.join(file_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    base_name, ext = os.path.splitext(os.path.basename(file_path))

    for size_mb in list_of_size_mb:
        sample_size_bytes = int(size_mb * 1024 * 1024)
        segment = sample_random_segment(file_path, sample_size_bytes, seed=seed)

        output_filename = f"{base_name}_sample_{size_mb}mb{ext}"
        output_path = os.path.join(temp_dir, output_filename)

        with open(output_path, "w", encoding="utf-8", errors="ignore") as out_f:
            out_f.write(segment)

        print(f"Generated {output_path}")


def run_tokenizer_on_samples(text_file_path, vocab_path, merge_path):
    """Encode a text file with Tokenizer and compute bytes/token.

    Args:
        text_file_path: Path to the text file to encode (can be a sampled file).
        vocab_path: Path to vocab JSON file.
        merge_path: Path to merges TXT file.

    Returns:
        (tokenizer, compression_ratio): The instantiated Tokenizer and the
        compression ratio measured as bytes per token (file_bytes / num_tokens).
        If no tokens are produced, compression_ratio is None.
    """
    tokenizer = Tokenizer.from_files(vocab_path, merge_path, ["<|endoftext|>"])

    # Read file contents (text) for encoding
    with open(text_file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Total bytes on disk for the file
    total_bytes = os.path.getsize(text_file_path)

    # Encode using the tokenizer
    token_ids = tokenizer.encode(text)
    num_tokens = len(token_ids)

    compression_ratio = None
    if num_tokens > 0:
        compression_ratio = total_bytes / num_tokens

    return tokenizer, compression_ratio


if __name__ == "__main__":
    dirpath = os.path.dirname(os.path.abspath(__file__))
    owt_text_filepath = os.path.join(dirpath, "data", "owt_train.txt")
    tiny_story_text_filepath = os.path.join(dirpath, "data", "TinyStoriesV2-GPT4-train.txt")
    sizes = [0.2, 0.5, 1, 2, 5, 10, 20, 35, 50, 100]
    #  generate sample files for tokenizer test
    generate_multiple_files(owt_text_filepath, sizes, seed=42)
    generate_multiple_files(tiny_story_text_filepath, sizes, seed=42)

