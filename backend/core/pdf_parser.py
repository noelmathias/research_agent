import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
from shared.schemas import PDFParseResult


class PDFParser:
    """
    PyMuPDF-based PDF parser.
    Extracts text page by page with metadata preservation.
    """

    @staticmethod
    def parse(file_path: str | Path) -> PDFParseResult:
        """
        Parse a PDF file and extract text from all pages.

        Returns PDFParseResult with per-page text and metadata.
        Raises FileNotFoundError if path doesn't exist.
        Raises ValueError if file is not a valid PDF.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        pages: List[Dict[str, Any]] = []
        total_chars = 0

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            raise ValueError(f"Could not open PDF: {e}")

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text — "text" mode preserves layout better than "blocks"
            text = page.get_text("text").strip()

            # Normalize whitespace runs while keeping paragraph breaks
            import re
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

            char_count = len(text)
            word_count = len(text.split()) if text else 0
            total_chars += char_count

            pages.append(
                {
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": char_count,
                    "word_count": word_count,
                }
            )

        doc.close()

        return PDFParseResult(
            filename=path.name,
            total_pages=len(pages),
            total_chars=total_chars,
            pages=pages,
        )

    @staticmethod
    def extract_full_text(file_path: str | Path) -> str:
        """Convenience: return all pages joined as single string."""
        result = PDFParser.parse(file_path)
        return "\n\n".join(
            p["text"] for p in result.pages if p["text"]
        )