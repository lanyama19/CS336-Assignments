import json
import regex
from typing import Dict, List, Tuple, Iterable, Iterator, Optional


class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_rank = {merge: rank for rank, merge in enumerate(merges)}
        self.regex = regex.compile(self.PAT)
        self.special_token_to_id = self._setup_special_tokens(special_tokens or [])
    
    def _setup_special_tokens(self, special_tokens: List[str]) -> Dict[str, int]:
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
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None) -> 'Tokenizer':
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    @staticmethod
    def _load_vocab(filepath: str) -> Dict[int, bytes]:
        with open(filepath, 'r') as f:
            vocab_json = json.load(f)
        return {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in vocab_json.items()}
    
    @staticmethod
    def _load_merges(filepath: str) -> List[Tuple[bytes, bytes]]:
        merges = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip() and len(line.split()) == 2:
                    hex1, hex2 = line.split()
                    merges.append((bytes.fromhex(hex1), bytes.fromhex(hex2)))
        return merges
    
    def _pre_tokenize(self, text: str) -> List[str]:
        return [match.group() for match in self.regex.finditer(text)]
    
    def _encode_pre_token(self, pre_token: str) -> List[int]:
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
        best_rank, best_merge = float('inf'), None
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merge_rank and self.merge_rank[pair] < best_rank:
                best_rank, best_merge = self.merge_rank[pair], (i, pair)
        return best_merge
    
    def _apply_merge(self, tokens: List[bytes], merge: Tuple[int, Tuple[bytes, bytes]]) -> List[bytes]:
        i, (token1, token2) = merge
        return tokens[:i] + [token1 + token2] + tokens[i+2:]
    
    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        
        if self.special_token_to_id:
            parts = self._split_on_special_tokens(text)
            return self._encode_parts(parts)
        
        return self._encode_text(text)
    
    def _split_on_special_tokens(self, text: str) -> List[Tuple[str, str]]:
        positions = []
        for token in self.special_token_to_id:
            start = 0
            while True:
                pos = text.find(token, start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(token), token))
                start = pos + 1
        
        positions.sort()
        parts = []
        last_end = 0
        
        for start, end, token in positions:
            if start > last_end:
                parts.append(('text', text[last_end:start]))
            parts.append(('special', token))
            last_end = end
        
        if last_end < len(text):
            parts.append(('text', text[last_end:]))
        
        return parts
    
    def _encode_parts(self, parts: List[Tuple[str, str]]) -> List[int]:
        token_ids = []
        for part_type, content in parts:
            if part_type == 'text':
                token_ids.extend(self._encode_text(content))
            else:
                token_ids.append(self.special_token_to_id[content])
        return token_ids
    
    def _encode_text(self, text: str) -> List[int]:
        token_ids = []
        for pre_token in self._pre_tokenize(text):
            token_ids.extend(self._encode_pre_token(pre_token))
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        
        all_bytes = []
        for token_id in ids:
            all_bytes.extend(self.vocab.get(token_id, b'\ufffd'))
        
        return bytes(all_bytes).decode('utf-8', errors='replace')
