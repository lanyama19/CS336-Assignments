import random
import os
import csv
import time
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
        (tokenizer, compression_ratio, throughput_bps):
        - compression_ratio: bytes per token (file_bytes / num_tokens), or None if 0 tokens.
        - throughput_bps: encoding throughput in bytes/second, or None if undeterminable.
    """
    tokenizer = Tokenizer.from_files(vocab_path, merge_path, ["<|endoftext|>"])

    # Read file contents (text) for encoding
    with open(text_file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Total bytes on disk for the file
    total_bytes = os.path.getsize(text_file_path)

    # Encode using the tokenizer (time the encoding only)
    start = time.perf_counter()
    token_ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    num_tokens = len(token_ids)

    compression_ratio = None
    if num_tokens > 0:
        compression_ratio = total_bytes / num_tokens

    throughput_bps = None
    if elapsed > 0:
        throughput_bps = total_bytes / elapsed

    return tokenizer, compression_ratio, throughput_bps


if __name__ == "__main__":
    #-------run the following for sampling test from raw data files-----------------------------------------------------
    # dirpath = os.path.dirname(os.path.abspath(__file__))
    # owt_text_filepath = os.path.join(dirpath, "data", "owt_train.txt")
    # tiny_story_text_filepath = os.path.join(dirpath, "data", "TinyStoriesV2-GPT4-train.txt")
    # sizes = [0.2, 0.5, 1, 2, 5, 10, 20, 35, 50, 100]
    # #  generate sample files for tokenizer test
    # generate_multiple_files(owt_text_filepath, sizes, seed=42)
    # generate_multiple_files(tiny_story_text_filepath, sizes, seed=42)

    # Iterate over data/temp files and evaluate compression (bytes/token)
    # using two tokenizer configs.



    dirpath = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(dirpath, "data", "temp")

    configs = [
        {
            "name": "bpe32k_owt",
            "vocab": os.path.join(dirpath, "data", "bpe_out_32000_1754934952", "vocab.json"),
            "merges": os.path.join(dirpath, "data", "bpe_out_32000_1754934952", "merges_hex.txt"),
        },
        {
            "name": "bpe10k_tinystory",
            "vocab": os.path.join(dirpath, "data", "bpe_out_10000_1754931137", "vocab.json"),
            "merges": os.path.join(dirpath, "data", "bpe_out_10000_1754931137", "merges_hex.txt"),
        },
    ]

    if not os.path.isdir(temp_dir):
        raise SystemExit(f"Missing directory: {temp_dir}")

    files = [
        os.path.join(temp_dir, f)
        for f in sorted(os.listdir(temp_dir))
        if os.path.isfile(os.path.join(temp_dir, f))
    ]

    if not files:
        raise SystemExit(f"No files found in {temp_dir}")

    print("Encoding files in data/temp and computing bytes/token:\n")
    results = []  # collect rows for CSV
    for cfg in configs:
        vocab_path = cfg["vocab"]
        merges_path = cfg["merges"]
        print(f"== Config: {cfg['name']} ==")
        for path in files:
            try:
                _, ratio, tput = run_tokenizer_on_samples(path, vocab_path, merges_path)
                size_b = os.path.getsize(path)
                if ratio is None:
                    ratio_str = "N/A (0 tokens)"
                else:
                    ratio_str = f"{ratio:.6f}"
                if tput is None:
                    tput_str = "N/A"
                else:
                    tput_str = f"{tput:.2f}"
                results.append([
                    cfg['name'], os.path.basename(path), size_b,
                    ("" if ratio is None else f"{ratio:.6f}"),
                    ("" if tput is None else f"{tput:.2f}")
                ])
                print(f"{os.path.basename(path)}\t{size_b} bytes\t{ratio_str} bytes/token\t{tput_str} bytes/s")
            except Exception as e:
                print(f"{os.path.basename(path)}\tERROR: {e}")
        print()

    # Write results to CSV in the temp directory
    out_csv = os.path.join(temp_dir, "compression_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "filename", "size_bytes", "bytes_per_token", "throughput_bytes_per_sec"])
        writer.writerows(results)
    print(f"Saved CSV: {out_csv}")

    
