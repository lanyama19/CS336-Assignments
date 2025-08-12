# pip install regex tqdm
import os, heapq, json, time, multiprocessing as mp
from typing import BinaryIO, List, Tuple, Dict, Set
from collections import Counter, defaultdict
import regex as re



SPECIAL = "<|endoftext|>"
SPECIAL_BYTES = SPECIAL.encode("utf-8")

# =============== CS336 helper ===============
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int, # This parameter is now less critical for chunking logic, but still passed
    split_special_token: bytes, # b'\n'
    max_chunk_bytes: int | None = None, # 10MB
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    boundaries = [0] # Start at the beginning of the file
    current_read_pos = 0 # Where we are currently reading from
    
    # Ensure a default max_chunk_bytes if not provided or invalid
    if max_chunk_bytes is None or max_chunk_bytes <= 0:
        max_chunk_bytes = 10 * 1024 * 1024 # Default to 10MB

    while current_read_pos < file_size:
        # Calculate the target end for the current chunk
        target_chunk_end = min(file_size, current_read_pos + max_chunk_bytes)
        
        # Read a segment that's at most max_chunk_bytes long, but could be longer if searching for newline
        # Read a bit more than max_chunk_bytes to find a newline if possible
        read_ahead_size = min(file_size - current_read_pos, max_chunk_bytes + 4096) # Read max_chunk_bytes + a buffer

        file.seek(current_read_pos)
        segment = file.read(read_ahead_size)

        if not segment: # End of file
            break

        # Try to find a newline within this segment
        newline_pos_in_segment = segment.rfind(split_special_token)

        if newline_pos_in_segment != -1: # Found a newline
            # The boundary is current_read_pos + position of newline + length of newline token
            boundary_candidate = current_read_pos + newline_pos_in_segment + len(split_special_token)
            
            # If this boundary candidate is within the target_chunk_end, use it.
            # Otherwise, if it overshoots too much, just cut at target_chunk_end.
            if boundary_candidate <= target_chunk_end + len(split_special_token) * 2: # Allow a small overshoot for newline
                boundaries.append(boundary_candidate)
                current_read_pos = boundary_candidate
            else:
                # Newline is too far, force a cut at target_chunk_end
                boundaries.append(target_chunk_end)
                current_read_pos = target_chunk_end
        else:
            # No newline found in the read_ahead_size segment.
            # This means we have a very long line. Force a cut at target_chunk_end.
            boundaries.append(target_chunk_end)
            current_read_pos = target_chunk_end

    # Ensure the last boundary is the end of the file
    if boundaries[-1] != file_size:
        boundaries.append(file_size)

    return sorted(list(set(boundaries))) # Sort and unique in case of overlaps/duplicates
# ============================================

# GPT-style pretokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# ---------- worker ----------
def _init_worker(pat_str: str):
    global RE_PAT, RE_SPECIAL
    RE_PAT = re.compile(pat_str)
    RE_SPECIAL = re.compile(r'(' + re.escape(SPECIAL) + r')')

def _count_chunk(path: str, start: int, end: int, special_tokens: List[str] | None = None, encoding="utf-8") -> Counter:
    cnt = Counter()
    with open(path, "rb") as fb:
        fb.seek(start)
        data = fb.read(end - start)
    text = data.decode(encoding, errors="ignore")
    # 统一行尾：将 CRLF/CR 规范化为 LF，避免将 '\r' 作为独立空白 token 计入
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # If we have special tokens, handle them; otherwise use original logic
    if special_tokens:
        # Create pattern for special tokens
        special_pattern = '|'.join(re.escape(token) for token in special_tokens)
        re_special = re.compile(f'({special_pattern})')
        special_set = set(special_tokens)
        
        # Split text by special tokens
        for seg in re_special.split(text):
            if seg in special_set:
                # This is a special token
                cnt[tuple(seg.encode("utf-8"))] += 1
            elif seg:
                # This is regular text, tokenize normally
                for m in RE_PAT.finditer(seg):
                    piece = m.group()
                    b = piece.encode("utf-8")
                    if b:
                        cnt[tuple(b)] += 1
    else:
        # Original logic - no special token handling
        for m in RE_PAT.finditer(text):
            piece = m.group()
            b = piece.encode("utf-8")
            if b:
                cnt[tuple(b)] += 1
    return cnt

def _count_chunk_star(args):
    return _count_chunk(*args)

def parallel_count_pieces(input_path: str, workers: int | None = None, max_chunk_bytes: int | None = None, special_tokens: List[str] | None = None) -> Counter:
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, workers, b'\n')
    starts, ends = boundaries[:-1], boundaries[1:]
    tasks_args = [(input_path, s, e, special_tokens) for s, e in zip(starts, ends)]

    total = Counter()
    if workers == 1:
        _init_worker(PAT)
        for args in tasks_args:
            total.update(_count_chunk_star(args))
        return total

    with mp.Pool(processes=workers, initializer=_init_worker, initargs=(PAT,)) as pool:
        it = pool.imap_unordered(_count_chunk_star, tasks_args, chunksize=1)
        for c in it:
            total.update(c)
    return total

# ---------- BPE core ----------
def seq_to_segments(seq: Tuple[int, ...]) -> List[bytes]:
    return [bytes([b]) for b in seq]

def segments_to_pairs(segs: List[bytes]) -> Counter:
    c = Counter()
    for i in range(len(segs)-1):
        c[(segs[i], segs[i+1])] += 1
    return c

def rebuild_indices_seg(corpus_segs: List[List[bytes]], freqs: List[int]):
    pair_count = defaultdict(int)
    occurs_in  = defaultdict(set)
    for wid, (segs, f) in enumerate(zip(corpus_segs, freqs)):
        if f == 0 or len(segs) < 2:
            continue
        local = segments_to_pairs(segs)
        for p, n in local.items():
            pair_count[p] += n * f
            occurs_in[p].add(wid)
    return pair_count, occurs_in

def apply_merge_in_segments(segs: List[bytes], X: bytes, Y: bytes, Z: bytes) -> bool:
    if len(segs) < 2: return False
    out, i, n, changed = [], 0, len(segs), False
    while i < n:
        if i < n-1 and segs[i]==X and segs[i+1]==Y:
            out.append(Z); i += 2; changed = True
        else:
            out.append(segs[i]); i += 1
    if changed: segs[:] = out
    return changed

def train_bpe_parallel(input_path: str,
                       vocab_size: int,
                       special_tokens: List[str] | None = None,
                       workers: int | None = None,
                       max_chunk_bytes: int | None = None):
    special_tokens = special_tokens or []
    special_set: Set[str] = set(special_tokens)

    t0 = time.perf_counter()
    counter = parallel_count_pieces(input_path, workers=workers, max_chunk_bytes=max_chunk_bytes, special_tokens=special_tokens if special_tokens else None)
    t1 = time.perf_counter()

    if special_tokens:
        # Separate special tokens from regular sequences for BPE processing
        special_token_bytes = set()
        for token in special_tokens:
            special_token_bytes.add(tuple(token.encode("utf-8")))
        
        # Only process non-special-token sequences with BPE
        corpus_seqs = []
        corpus_freqs = []
        special_counts = {}
        
        for seq, freq in counter.items():
            if seq in special_token_bytes:
                # This is a special token, store separately
                special_counts[seq] = freq
            else:
                # Regular sequence, add to BPE processing
                corpus_seqs.append(seq)
                corpus_freqs.append(freq)
        
        corpus_segs: List[List[bytes]] = [seq_to_segments(seq) for seq in corpus_seqs]
        freqs: List[int] = corpus_freqs
    else:
        # Original behavior when no special tokens
        corpus_segs: List[List[bytes]] = [seq_to_segments(seq) for seq in counter.keys()]
        freqs: List[int]               = [counter[seq] for seq in counter.keys()]

    symbols: Set[bytes] = set(bytes([i]) for i in range(256))
    max_merges = max(0, vocab_size - (len(symbols) + len(special_set)))

    pair_count, occurs_in = rebuild_indices_seg(corpus_segs, freqs)
    t2 = time.perf_counter()

    merges: List[Tuple[bytes, bytes]] = []
    for _ in range(max_merges):
            if not pair_count:
                break
            # Find the best pair: max count, then lexicographic order for tie-breaking
            # Use the standard BPE tie-breaking: highest count, then lexicographically largest pair
            best = max(pair_count, key=lambda p: (pair_count[p], p))
            if pair_count[best] <= 0:
                break

            X, Y = best
            Z = X + Y
            affected = list(occurs_in.get(best, ()))

            for wid in affected:
                if len(corpus_segs[wid]) < 2:
                    continue
                old_pairs = segments_to_pairs(corpus_segs[wid])
                if not apply_merge_in_segments(corpus_segs[wid], X, Y, Z):
                    continue
                new_pairs = segments_to_pairs(corpus_segs[wid])

                for p in old_pairs.keys() | new_pairs.keys():
                    delta = new_pairs.get(p, 0) - old_pairs.get(p, 0)
                    if delta == 0:
                        continue
                    pair_count[p] = pair_count.get(p, 0) + delta * freqs[wid]
                    if pair_count[p] <= 0:
                        if p in pair_count:
                            del pair_count[p]
                    if delta > 0:
                        occurs_in[p].add(wid)
                    elif delta < 0:
                        if new_pairs.get(p, 0) == 0 and wid in occurs_in.get(p, set()):
                            occurs_in[p].discard(wid)

            if best in occurs_in:
                occurs_in[best].clear()
            if best in pair_count:
                del pair_count[best]  # Remove instead of setting to 0
            merges.append((X, Y))
            if len(symbols) >= vocab_size - len(special_set):
                break
    t3 = time.perf_counter()

    id2tok: Dict[int, bytes] = {}
    nid = 0
    for i in range(256):
        id2tok[nid] = bytes([i]); nid += 1
    seen_added: Set[bytes] = set()
    for X, Y in merges:
        Z = X + Y
        if Z not in seen_added:
            if nid >= vocab_size - len(special_set): break
            id2tok[nid] = Z
            seen_added.add(Z)
            nid += 1
    for s in special_tokens:
        if nid >= vocab_size: break
        id2tok[nid] = s.encode("utf-8"); nid += 1
    t4 = time.perf_counter()

    # print timing
    print(f"[Timing] pretokenize+count: {t1 - t0:.3f}s | build_indices: {t2 - t1:.3f}s | merges: {t3 - t2:.3f}s | build_vocab: {t4 - t3:.3f}s | total: {t4 - t0:.3f}s")
    return id2tok, merges

# ---------- saving ----------
def _bytes_to_hex(b: bytes) -> str:
    return b.hex()

def save_artifacts(out_dir: str, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], meta: Dict):
    os.makedirs(out_dir, exist_ok=True)
    # vocab.json: {id: "hex"}
    vocab_json = {int(k): _bytes_to_hex(v) for k, v in vocab.items()}
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    # merges_hex.txt: "hex1 hex2"
    with open(os.path.join(out_dir, "merges_hex.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_bytes_to_hex(a)} {_bytes_to_hex(b)}\n")
    # meta.json
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_dir}")

# ------------- example -------------
if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).resolve().parent
    path = here / "data" / "owt_train.txt"
    path = str(path)
    vocab_size = 32000
    start = time.perf_counter()
    # --- Memory logging point 1 (before training) ---
    # import psutil
    # import os
    # process = psutil.Process(os.getpid())
    # print(f"Memory usage before training: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    # -------------------------------------------------
    vocab, merges = train_bpe_parallel(
        input_path=path,
        vocab_size=vocab_size,
        special_tokens=[SPECIAL],
        workers=8,   # None => CPU-1
        max_chunk_bytes=10 * 1024 * 1024 # 10 MB
    )
    end = time.perf_counter()
    # --- Memory logging point 2 (after training) ---
    # import psutil
    # import os
    # process = psutil.Process(os.getpid())
    # print(f"Memory usage after training: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    # ------------------------------------------------

    print("vocab size:", len(vocab))
    print("first 5 ids:", {k: vocab[k] for k in range(5)})
    print("first 5 merges:", merges[:5])

    # save artifacts next to data
    run_dir = os.path.join(os.path.dirname(path), f"bpe_out_{vocab_size}_{int(time.time())}")
    meta = {
        "input_path": path,
        "vocab_size": vocab_size,
        "num_merges": len(merges),
        "total_time_sec": round(end - start, 3),
        "python": os.sys.version.split()[0],
    }
    save_artifacts(run_dir, vocab, merges, meta)
