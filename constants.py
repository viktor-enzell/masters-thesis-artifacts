kb_vocab = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "|": 4,
    "T": 5,
    "E": 6,
    "A": 7,
    "N": 8,
    "R": 9,
    "S": 10,
    "I": 11,
    "L": 12,
    "D": 13,
    "O": 14,
    "M": 15,
    "K": 16,
    "G": 17,
    "U": 18,
    "V": 19,
    "F": 20,
    "H": 21,
    "Ä": 22,
    "Å": 23,
    "P": 24,
    "Ö": 25,
    "B": 26,
    "J": 27,
    "C": 28,
    "Y": 29,
    "X": 30,
    "W": 31,
    "Z": 32,
    "É": 33,
    "Q": 34,
    "8": 35,
    "2": 36,
    "5": 37,
    "9": 38,
    "1": 39,
    "6": 40,
    "7": 41,
    "3": 42,
    "4": 43,
    "0": 44,
    "'": 45,
}

kb_vocab_keys = list(kb_vocab.keys())[5:]

CHARS_TO_IGNORE = f'[^{"".join(kb_vocab_keys)}\ ]'
CHARS_TO_BLANK = r"[\t]"
REMOVE_BOLD = r"<b>|</b>"

ACOUSTIC_MODEL_NAME = "KBLab/wav2vec2-large-voxrex-swedish"

DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 1.5
DEFAULT_HOTWORD_WEIGHT = 10.0
