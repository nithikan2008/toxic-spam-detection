import re
import emoji
import random
import pandas as pd



def decode_obfuscation(text:str) -> str:
    text = text.lower()

    text = (
        text.replace("0", "o")
            .replace("1", "i")
            .replace("3", "e")
            .replace("4", "a")
            .replace("5", "s")
            .replace("7", "t")
    )

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text

def map_emojis(text: str) -> str:
    # Convert emoji → descriptive word
    return emoji.demojize(text, delimiters=(" ", " "))

def tokenize(text: str):
    # Simple whitespace + punctuation split
    return [tok for tok in re.split(r"\W+", text) if tok]

def preprocess_text(text: str) -> str:
    text = decode_obfuscation(text)
    text = map_emojis(text)
    tokens = tokenize(text)
    return " ".join(tokens)

if __name__ == "__main__":
    samples = [
        "fr33 iph0ne!!! 😂😂",
        "you 1di0t 🤬",
        "c@sh.app now!!",
        "nooooooo wayyyy 😭💀",
    ]

    for s in samples:
        print(f"RAW: {s}")
        print(f"PROC: {preprocess_text(s)}\n")
