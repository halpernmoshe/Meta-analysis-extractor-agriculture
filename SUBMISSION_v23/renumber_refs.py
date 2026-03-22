#!/usr/bin/env python3
"""Renumber AMA references in PAPER_FINAL_v23.md to match order of first appearance."""
import re
from pathlib import Path

MD_FILE = Path(__file__).parent / "PAPER_FINAL_v23.md"

# Mapping: old number -> new number
# Refs 1-16 stay the same.
# Refs 32,33,34 appear on line 58 (Section 1.3) before refs 17-31.
# So 32->17, 33->18, 34->19, then old 17->20, 18->21, ..., 31->34.
REMAP = {}
for i in range(1, 17):
    REMAP[i] = i  # 1-16 unchanged

REMAP[32] = 17  # Gartlehner 2024
REMAP[33] = 18  # Khraisha 2024
REMAP[34] = 19  # Li L. 2025

for old in range(17, 32):  # 17->20, 18->21, ..., 31->34
    REMAP[old] = old + 3

def renumber():
    text = MD_FILE.read_text(encoding='utf-8')

    # Step 1: Find reference list and extract entries keyed by old number
    ref_start = text.find('# References\n')
    ref_end = text.find('\n---\n\n## Declarations')
    if ref_start == -1 or ref_end == -1:
        print("ERROR: Could not find reference block boundaries")
        return

    ref_block = text[ref_start:ref_end]
    ref_entries = {}
    for match in re.finditer(r'^(\d+)\.\s+(.*)', ref_block, re.MULTILINE):
        old_num = int(match.group(1))
        ref_text = match.group(2)
        ref_entries[old_num] = ref_text

    print(f"Found {len(ref_entries)} references")

    # Step 2: Replace all ^N^ citations in body text with temporary markers
    # Also handle ^N,M^ compound citations
    body_text = text[:ref_start]

    # Replace compound citations first (e.g., ^14,32^)
    def replace_compound(m):
        inner = m.group(1)
        nums = [n.strip() for n in inner.split(',')]
        new_nums = []
        for n in nums:
            old = int(n)
            new = REMAP.get(old, old)
            new_nums.append(str(new))
        # Sort the numbers in the compound citation
        new_nums.sort(key=int)
        return '^' + ','.join(new_nums) + '^'

    # First, handle compound citations like ^14,32^
    body_text = re.sub(r'\^(\d+(?:,\s*\d+)+)\^', replace_compound, body_text)

    # Then handle single citations ^N^ — use temp markers to avoid collisions
    for old_num in sorted(REMAP.keys(), reverse=True):
        new_num = REMAP[old_num]
        if old_num != new_num:
            body_text = body_text.replace(f'^{old_num}^', f'^TEMP{new_num}^')

    # Convert temp markers back
    body_text = re.sub(r'\^TEMP(\d+)\^', r'^\1^', body_text)

    # Step 3: Rebuild reference list with new numbering
    new_refs = ['# References\n']
    for new_num in sorted(REMAP.values()):
        # Find which old number maps to this new number
        old_num = [k for k, v in REMAP.items() if v == new_num][0]
        if old_num in ref_entries:
            new_refs.append(f'{new_num}. {ref_entries[old_num]}')
        else:
            print(f"WARNING: No reference entry for old #{old_num} (new #{new_num})")
    new_refs.append('')

    # Step 4: Also renumber citations in Declarations section (after references)
    decl_text = text[ref_end:]
    for old_num in sorted(REMAP.keys(), reverse=True):
        new_num = REMAP[old_num]
        if old_num != new_num:
            decl_text = decl_text.replace(f'^{old_num}^', f'^TEMP{new_num}^')
    decl_text = re.sub(r'\^TEMP(\d+)\^', r'^\1^', decl_text)

    # Reassemble
    text = body_text + '\n'.join(new_refs) + '\n' + decl_text

    MD_FILE.write_text(text, encoding='utf-8')
    print("Reference renumbering complete.")
    print("Mapping applied:")
    for old in sorted(REMAP.keys()):
        new = REMAP[old]
        if old != new:
            print(f"  {old} -> {new}")

if __name__ == '__main__':
    renumber()
