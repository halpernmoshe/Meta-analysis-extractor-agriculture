#!/usr/bin/env python3
"""Convert Markdown files to PDF using fpdf2."""
import re
from pathlib import Path
from fpdf import FPDF

# Unicode -> ASCII replacements for Helvetica compatibility
UNICODE_MAP = {
    '\u2014': '--',   # em dash
    '\u2013': '-',    # en dash
    '\u2019': "'",    # right single quote
    '\u2018': "'",    # left single quote
    '\u201c': '"',    # left double quote
    '\u201d': '"',    # right double quote
    '\u2022': '-',    # bullet
    '\u00b1': '+-',   # plus-minus
    '\u2264': '<=',   # less than or equal
    '\u2265': '>=',   # greater than or equal
    '\u00d7': 'x',    # multiplication sign
    '\u2026': '...',  # ellipsis
    '\u03b1': 'alpha',
    '\u03b2': 'beta',
    '\u0394': 'Delta',
}


def sanitize(text):
    """Replace Unicode chars that Helvetica can't render."""
    for u, a in UNICODE_MAP.items():
        text = text.replace(u, a)
    # Strip any remaining non-latin1 chars
    return text.encode('latin-1', errors='replace').decode('latin-1')


class MarkdownPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', new_x="RIGHT", new_y="TOP", align='C')


def break_long_words(text, max_chars=70):
    """Insert soft breaks in words longer than max_chars."""
    words = text.split(' ')
    result = []
    for w in words:
        if len(w) > max_chars:
            # Break at slashes, dots, underscores, hyphens
            broken = re.sub(r'([/._-])', r'\1 ', w)
            # If still too long, force-break every max_chars
            parts = broken.split(' ')
            final_parts = []
            for p in parts:
                while len(p) > max_chars:
                    final_parts.append(p[:max_chars])
                    p = p[max_chars:]
                final_parts.append(p)
            result.append(' '.join(final_parts))
        else:
            result.append(w)
    return ' '.join(result)


def clean_markdown(text):
    """Strip markdown formatting to plain text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Convert ^N^ superscript citations to [N]
    text = re.sub(r'\^(\d[\d,]*)\^', r'[\1]', text)
    return text


def md_to_pdf(md_path, pdf_path):
    text = Path(md_path).read_text(encoding='utf-8')
    lines = text.split('\n')

    pdf = MarkdownPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    pdf.set_font('Helvetica', '', 10)

    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Code block toggle
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            if in_code_block:
                pdf.ln(2)
            continue

        # Inside code block
        if in_code_block:
            pdf.set_font('Courier', '', 7)
            display = sanitize(line[:110])
            pdf.cell(0, 4, display, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Helvetica', '', 10)
            continue

        if not stripped:
            pdf.ln(3)
            continue

        if stripped == '---':
            pdf.ln(2)
            continue

        # Skip table formatting/separator lines (e.g., |---|---|)
        if re.match(r'^\|[\s\-|:]+\|?$', stripped):
            continue
        if stripped.startswith('|--') or stripped.startswith('|:--'):
            continue

        # Table rows
        if stripped.startswith('|'):
            pdf.set_font('Helvetica', '', 7)
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            clean_row = '  |  '.join(cells)
            clean_row = clean_markdown(clean_row)
            clean_row = sanitize(clean_row)
            pdf.cell(0, 4, clean_row[:140], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Helvetica', '', 10)
            continue

        # Headings
        if stripped.startswith('# '):
            pdf.ln(4)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.multi_cell(0, 7, sanitize(clean_markdown(stripped[2:])))
            pdf.ln(2)
            pdf.set_font('Helvetica', '', 10)
            continue
        if stripped.startswith('## '):
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.multi_cell(0, 6, sanitize(clean_markdown(stripped[3:])))
            pdf.ln(1)
            pdf.set_font('Helvetica', '', 10)
            continue
        if stripped.startswith('### '):
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.multi_cell(0, 5, sanitize(clean_markdown(stripped[4:])))
            pdf.ln(1)
            pdf.set_font('Helvetica', '', 10)
            continue
        if stripped.startswith('#### '):
            pdf.ln(1)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.multi_cell(0, 5, sanitize(clean_markdown(stripped[5:])))
            pdf.set_font('Helvetica', '', 10)
            continue

        # Clean markdown formatting
        clean = clean_markdown(stripped)
        clean = sanitize(clean)

        # Bullet points
        if clean.startswith('- '):
            clean = '  - ' + clean[2:]

        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+(.*)', clean)
        if num_match:
            clean = f'  {num_match.group(1)}. {num_match.group(2)}'

        clean = break_long_words(clean)
        pdf.multi_cell(0, 5, clean)

    pdf.output(str(pdf_path))
    size = Path(pdf_path).stat().st_size
    print(f"Created {pdf_path} ({size:,} bytes)")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    out_dir = script_dir / 'SUBMISSION_CLEAN'
    out_dir.mkdir(exist_ok=True)

    md_to_pdf(script_dir / 'PAPER_FINAL_v23.md', out_dir / 'Halpern_Manuscript_2026.pdf')
    md_to_pdf(script_dir / 'COVER_LETTER.md', out_dir / 'Cover_Letter.pdf')
    md_to_pdf(script_dir / 'CODE_AVAILABILITY.md', out_dir / 'Code_Availability.pdf')
