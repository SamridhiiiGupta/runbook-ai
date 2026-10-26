from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """One piece of a document, plus where it came from."""
    source: str   # file name, e.g. "vpn_access.md"
    section: str  # heading, e.g. "VPN troubleshooting"
    text: str     # the actual content


def split_into_chunks(text: str, source: str) -> list[Chunk]:
    """Cut one document into chunks, one per '## ' heading."""
    chunks = []
    section = "Overview"
    buffer = []

    def save():
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(Chunk(source=source, section=section, text=body))

    for line in text.splitlines():
        if line.startswith("## "):
            save()                      # finish the previous section
            section = line[3:].strip()  # start a new one
            buffer = []
        else:
            buffer.append(line)
    save()                              # don't forget the last section
    return chunks


def load_runbooks(folder: str) -> list[Chunk]:
    """Read every .md file in a folder and chunk them all."""
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(split_into_chunks(text, source=path.name))
    return chunks