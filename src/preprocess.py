import numpy as np
import torch

class LabelEncoder:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char_to_int = {c: i + 1 for i, c in enumerate(alphabet)}
        self.int_to_char = {i + 1: c for i, c in enumerate(alphabet)}
        self.vocab_size = len(alphabet) + 1  # +1 for padding (0)

    def encode(self, s, max_len=100):
        # Truncate or pad to max_len
        if isinstance(s, float): # Handle nan strings if any
            s = ""
        s = str(s)
        encoded = [self.char_to_int.get(c, 0) for c in s] # 0 for unknown
        
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded = encoded + [0] * (max_len - len(encoded))
            
        return np.array(encoded)

# Common alphabets
# DeepDTA used:
SMILES_ALPHABET = {"#", "%", ")", "(", "+", "-", ".", "1", "0", "3", "2", "5", "4", "7", "6", "9", "8", "=", "A", "C", "B", "E", "D", "G", "F", "I", "H", "K", "M", "L", "O", "N", "P", "S", "R", "U", "T", "W", "V", "Y", "[", "]", "c", "e", "g", "i", "l", "o", "n", "s", "r", "@"}
TARGET_ALPHABET = {"A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"}

drug_encoder = LabelEncoder(sorted(list(SMILES_ALPHABET)))
target_encoder = LabelEncoder(sorted(list(TARGET_ALPHABET)))
