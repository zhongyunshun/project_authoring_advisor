"""Streamlit auto-discovers pages from this directory next to app.py.
The actual upload logic lives in ui/pages/upload_pdf.py — this is a thin
loader so we don't duplicate code.
"""

import runpy
from pathlib import Path

target = Path(__file__).resolve().parent.parent / "ui" / "pages" / "upload_pdf.py"
runpy.run_path(str(target), run_name="__main__")
