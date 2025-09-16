from __future__ import annotations
from typing import Optional
from pathlib import Path

def read_text_file(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")

def pdf_to_text(path: str) -> Optional[str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        return None

def load_cv(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".txt") or path_lower.endswith(".md"):
        return read_text_file(path)
    if path_lower.endswith(".pdf"):
        txt = pdf_to_text(path)
        if txt:
            return txt
        raise RuntimeError("Gagal ekstrak teks dari PDF. Install pypdf atau pastikan file tidak terenkripsi.")
    raise ValueError("Format CV tidak didukung. Gunakan .txt atau .pdf")
