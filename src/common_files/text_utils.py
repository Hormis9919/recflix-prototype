import re
from typing import List

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t)>1]
    return tokens