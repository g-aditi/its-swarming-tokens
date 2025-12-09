
import unicodedata
import string
import re

class TokenFilter:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.start_prefix_id = tokenizer.encode("«",add_special_tokens=False)[0]
        self.embedding_size = model.get_input_embeddings().weight.detach().size(0)
        self.START_PREFIX = "«"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.prefix_error = []
        
    def encode(self,s):
        s = self.START_PREFIX + s
        tokens = self.tokenizer.encode(s, add_special_tokens=False)
        if tokens[0] != self.start_prefix_id:
            self.prefix_error.append(tokens[0])
            return [0, 1]
        tokens = tokens[1:]
        return tokens

    def decode(self, tokens):
        tokens = [self.start_prefix_id] + tokens
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
        if decoded[0] == " ":  # e.g. mistral, but not llama2
            decoded = decoded[1:]
        assert decoded.startswith(
                self.START_PREFIX
            ), f"The decoded string {decoded!r} should start with the prefix for {tokens!r}"
        return decoded[1:]
    
    def token_classify(self, token_id):
        try:
            s = self.decode([token_id])
        except:
            return "UNDECODEABLE"
        
        if not self.is_roman_only(s):
            return "NON_ROMAN"

        id = self.encode(s)
        if id == [token_id]:
            if len(s) >= 3 and s[0] in "[<" and s[-1] in "]>" and any(c.isalpha() for c in s):
                return "SPECIAL"
            elif "�" in s:
                return "UNDECODEABLE"
            else:
                return "NORMAL"
        else:
            return "UNREACHABLE"
    
    def filter_token(self):
        skip_tokens = []
        for id in range(self.embedding_size):
            category = self.token_classify(id)
            if category != "NORMAL":
                skip_tokens.append(id)
        return skip_tokens
    
    def get_ith_token(self, id):
        s= self.decode([id])
        if ' ' in s:
            s = s.replace(' ', '▁')
        return s
    

    def is_roman_string(self, s):
        for ch in s:
            if ch in string.printable:
                continue
            return False
        return True
    
    def is_roman_only(self, s):
        if not re.search(r"[A-Za-z]", s):
            return False
        return all(32 <= ord(ch) < 127 for ch in s)