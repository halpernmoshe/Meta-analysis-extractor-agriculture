#!/usr/bin/env python3
"""Convert Markdown files to PDF using fpdf2."""
import re
from pathlib import Path
from fpdf import FPDF


class MarkdownPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', new_x="RIGHT", new_y="TOP", align='C')


def break_long_words(text, max_chars=60):
    """Insert soft breaks in words longer than max_chars."""
    words = text.split(' ')
    result = []
    for w in words:
        if len(w) > max_chars:
            # Break at slashes, dots, or every max_chars
            broken = re.sub(r'([/._-])', r'\1 ', w)
            result.append(broken)
        else:
            result.append(w)
    return ' '.join(result)


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

        # Inside code block - render as monospace
        if in_code_block:
            pdf.set_font('Courier', '', 8)
            # Truncate very long lines
            display = line[:100]
            pdf.cell(0, 4, display, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Helvetica', '', 10)
            continue

        if not stripped:
            pdf.ln(3)
            continue

        if stripped == '---':
            pdf.ln(2)
            continue

        # Skip table formatting lines
        if re.match(r'^\|[-\s|:]+\|$', stripped):
            continue

        # Table rows
        if stripped.startswith('|'):
            pdf.set_font('Helvetica', '', 8)
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            clean_row = '  |  '.join(cells)
            clean_row = re.sub(r'\*\*(.+?)\*\*', r'\1', clean_row)
            clean_row = re.sub(r'`(.+?)`', r'\1', clean_row)
            pdf.cell(0, 4, clean_row[:120], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Helvetica', '', 10)
            continue

        # Headings
        if stripped.startswith('# '):
            pdf.ln(4)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.multi_cell(0, 7, stripped[2:])
            pdf.ln(2)
            pdf.set_font('Helvetica', '', 10)
            continue
        if stripped.startswith('## '):
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.multi_cell(0, 6, stripped[3:])
            pdf.ln(1)
            pdf.set_font('Helvetica', '', 10)
            continue
        if stripped.startswith('### '):
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.multi_cell(0, 5, stripped[4:])
            pdf.ln(1)
            pdf.set_font('Helvetica', '', 10)
            continue

        # Clean markdown formatting
        clean = stripped
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
        clean = re.sub(r'\*(.+?)\*', r'\1', clean)
        clean = re.sub(r'\[(.+?)\]\((.+?)\)', r'\1 (\2)', clean)
        clean = re.sub(r'`(.+?)`', r'\1', clean)
        clean = clean.replace('\u2014', '--').replace('\u2013', '-')

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

    md_to_pdf(script_dir / 'COVER_LETTER.md', out_dir / 'Cover_Letter.pdf')
    md_to_pdf(script_dir / 'CODE_AVAILABILITY.md', out_dir / 'Code_Availability.pdf')
