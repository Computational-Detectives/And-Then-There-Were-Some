import re
from pathlib import Path

def strip_lines(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return re.sub(r"[^\S\n]*\n[^\S\n]*", "\n", text)


def main(path: str):
    with open('data/book/attwn_six_cleaned.md', 'w') as f:
        cleaned = strip_lines(path)
        f.write(cleaned)


if __name__ == '__main__':
    path = 'data/book/attwn_six.md'
    main(path)