import json
import regex
from typing import Dict, List, Tuple, Iterable, Iterator, Optional
import os


class Tokenizer:
    # GPT-2 pre-tokenization regex
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def __init__(self, vocab: Dict[int, bytes],
                 merges: List[Tuple[bytes, bytes]],
                 special_tokens: Optional[List[str]] = None):
        """
        Args:
            vocab: Mapping from token id → token bytes. Must contain all byte sequences
                produced by merging.
            merges: Ordered list of merges, where each item is a pair of byte sequences.
                Earlier items have higher priority (lower rank) during merging.
            special_tokens: Optional list of string special tokens that should be
                treated atomically and never split.
        """
        self.vocab = vocab
        # Inverse mapping for quick lookup during encoding
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        # Assign a rank to each merge (lower rank = higher priority)
        self.merge_rank = {merge: rank for rank, merge in enumerate(merges)}
        self.regex = regex.compile(self.PAT)
        # Register special tokens (append to vocab if missing)
        self.special_token_to_id = self._setup_special_tokens(special_tokens or [])

    @staticmethod
    def _build_gpt2_decoder() -> Dict[str, int]:
        bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        chars = [chr(n) for n in cs]
        enc = dict(zip(bs, chars))
        return {v: k for k, v in enc.items()}

    @staticmethod
    def _is_hex_string(s: str) -> bool:
        if not isinstance(s, str) or len(s) % 2 != 0:
            return False
        try:
            int(s, 16)
            return True
        except Exception:
            return False
    
    def _setup_special_tokens(self, special_tokens: List[str]) -> Dict[str, int]:
        """Ensure special tokens exist in the vocab and build a string→id mapping.
        If a special token's UTF-8 byte sequence is not in the vocab, append it.
        """
        special_token_to_id = {}
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id
            special_token_to_id[token] = self.bytes_to_id[token_bytes]
        return special_token_to_id
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> 'Tokenizer':
        """Construct a tokenizer from vocab/merges files (auto-detected format).

        Supports two formats:
          1) GPT-2 printable mapping files (as in tests/fixtures)
          2) Hex-based files (id → hex; merges are hex pairs per line)
        """
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    @staticmethod
    def _load_vocab(filepath: str) -> Dict[int, bytes]:
        """Load a vocabulary mapping id→bytes, auto-detecting GPT-2 printable vs hex formats."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # GPT-2 printable: values are ints
        if all(isinstance(v, int) for v in data.values()):
            decoder = Tokenizer._build_gpt2_decoder()
            return {v: bytes([decoder[ch] for ch in k]) for k, v in data.items()}

        # Hex: keys are numeric strings and values are hex strings
        if all(isinstance(k, str) and k.isdigit() for k in data.keys()) and \
           all(Tokenizer._is_hex_string(v) for v in data.values()):
            return {int(k): bytes.fromhex(v) for k, v in data.items()}

        raise ValueError('Unrecognized vocab format: expected GPT-2 printable or hex mapping')
    
    @staticmethod
    def _load_merges(filepath: str) -> List[Tuple[bytes, bytes]]:
        """Load merges, auto-detecting GPT-2 printable vs hex formats."""
        pairs: List[Tuple[str, str]] = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned = line.rstrip('\r\n')
                if cleaned and len(cleaned.split(' ')) == 2:
                    a, b = cleaned.split(' ')
                    pairs.append((a, b))

        # Hex format if all tokens look hex
        if pairs and all(Tokenizer._is_hex_string(a) and Tokenizer._is_hex_string(b) for a, b in pairs):
            return [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in pairs]

        # Otherwise treat as GPT-2 printable
        decoder = Tokenizer._build_gpt2_decoder()
        to_bytes = lambda s: bytes([decoder[ch] for ch in s])
        return [(to_bytes(a), to_bytes(b)) for a, b in pairs]
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Split text into GPT-2 style pre-tokens using the compiled regex."""
        return [match.group() for match in self.regex.finditer(text)]
    
    def _encode_pre_token(self, pre_token: str) -> List[int]:
        """Encode a single pre-token via byte-level BPE merges.

        - UTF-8 encode the pre-token and start from single-byte tokens
        - While possible, merge the best-ranked adjacent pair
        - Map merged byte sequences to token ids via the vocab
        """
        if not pre_token:
            return []

        tokens = [bytes([b]) for b in pre_token.encode('utf-8')]

        while True:
            best_merge = self._find_best_merge(tokens)
            if not best_merge:
                break
            tokens = self._apply_merge(tokens, best_merge)

        return [self.bytes_to_id[token] for token in tokens]
    
    def _find_best_merge(self, tokens: List[bytes]) -> Optional[Tuple[int, Tuple[bytes, bytes]]]:
        """Return the index and pair of the best-ranked adjacent merge, if any."""
        best_rank, best_merge = float('inf'), None
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merge_rank and self.merge_rank[pair] < best_rank:
                best_rank, best_merge = self.merge_rank[pair], (i, pair)
        return best_merge
    
    def _apply_merge(self, tokens: List[bytes], merge: Tuple[int, Tuple[bytes, bytes]]) -> List[bytes]:
        """Apply the chosen merge at index i by concatenating adjacent byte tokens."""
        i, (token1, token2) = merge
        return tokens[:i] + [token1 + token2] + tokens[i+2:]
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token ids, handling specials and BPE in one pass."""
        if not text:
            return []

        token_ids: List[int] = []
        if self.special_token_to_id:
            for part_type, content in self._split_on_special_tokens(text):
                if part_type == 'special':
                    token_ids.append(self.special_token_to_id[content])
                else:
                    for pre_token in self._pre_tokenize(content):
                        token_ids.extend(self._encode_pre_token(pre_token))
            return token_ids

        for pre_token in self._pre_tokenize(text):
            token_ids.extend(self._encode_pre_token(pre_token))
        return token_ids
    
    def _split_on_special_tokens(self, text: str) -> List[Tuple[str, str]]:
        """Greedy left-to-right longest-match split on special tokens.
        # CHANGED: Rewrote splitting to greedy longest-match to correctly
        # handle overlapping specials (e.g., "<|eot|><|eot|>") and preserve
        # special tokens atomically without being merged/split.
        Returns a list of tagged parts: ('text', segment) or ('special', token).
        Overlapping patterns are resolved by always picking the longest match first.
        """
        if not self.special_token_to_id:
            return [('text', text)] if text else []

        specials = sorted(self.special_token_to_id.keys(), key=len, reverse=True)
        i = 0
        parts: List[Tuple[str, str]] = []
        acc_start = 0
        n = len(text)

        while i < n:
            match_token = None
            for tok in specials:
                if text.startswith(tok, i):
                    match_token = tok
                    break
            if match_token is None:
                i += 1
                continue
            # Flush any preceding text
            if acc_start < i:
                parts.append(('text', text[acc_start:i]))
            # Append the special token
            parts.append(('special', match_token))
            i += len(match_token)
            acc_start = i

        if acc_start < n:
            parts.append(('text', text[acc_start:]))
        return parts

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token ids.
        # Concatenate the iterable and encode once to ensure exact
        # equivalence with whole-text encoding and match reference ids.
        # This avoids line-boundary artifacts from pre-tokenization/BPE.
        """
        full_text_parts: List[str] = []
        for text in iterable:
            full_text_parts.append(text)
        if not full_text_parts:
            return  # nothing to yield
        for _id in self.encode(''.join(full_text_parts)):
            yield _id
    
    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back into a string.
        # Added robust fallback for unknown ids using the UTF-8
        # replacement character bytes to avoid invalid sequences and keep
        # roundtrips stable across platforms.
        """
        if not ids:
            return ""

        all_bytes = bytearray()
        for token_id in ids:
            token_bytes = self.vocab.get(token_id)
            if token_bytes is None:
                # Use the UTF-8 bytes for the replacement character '�'
                token_bytes = b"\xef\xbf\xbd"
            all_bytes.extend(token_bytes)
        return bytes(all_bytes).decode('utf-8', errors='replace')


if __name__ == "__main__":
    # Smoke test using auto-detected loaders and GPT-2 fixtures.
    dirpath = os.path.dirname(os.path.abspath(__file__))
    fixtures_path = os.path.join(dirpath, "tests", "fixtures")
    VOCAB_PATH = os.path.join(fixtures_path, "gpt2_vocab.json")
    MERGES_PATH = os.path.join(fixtures_path, "gpt2_merges.txt")
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])
    sample = "Hello, this is CS-336"
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    print("Loaded fixtures. Roundtrip ok:", decoded == sample)
