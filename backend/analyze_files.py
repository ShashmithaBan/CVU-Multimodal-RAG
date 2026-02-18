"""Analyze OCR content to find actual file names in the video."""
import json
import re
from collections import Counter

with open('outputs/OPaLnMw2i_0_chunks_metadata.json') as f:
    data = json.load(f)

chunks = data['chunks']
print(f'Total chunks: {len(chunks)}')

# Find all potential file names
all_files = []
file_patterns = [
    r'(\w+\.(?:js|jsx|ts|tsx|css|json))',
    r'([A-Z][a-zA-Z]+\.js)',
    r'src/([^/\s]+)',
    r'components/([^/\s]+)',
]

for chunk in chunks:
    ocr = chunk.get('ocr_text', '')
    if not ocr:
        continue
    
    for pattern in file_patterns:
        matches = re.findall(pattern, ocr)
        all_files.extend(matches)

counter = Counter(all_files)
print('\nTop 30 detected file names:')
for f, count in counter.most_common(30):
    print(f'  {f}: {count} occurrences')

# Code patterns
print('\n--- Code Analysis ---')
code_chunks = [c for c in chunks if 'import' in c.get('ocr_text', '').lower() or 'export' in c.get('ocr_text', '').lower()]
print(f'Chunks with import/export: {len(code_chunks)}')

# Find components defined
components = []
functions = []
for chunk in chunks:
    ocr = chunk.get('ocr_text', '')
    if not ocr:
        continue
    
    # Component definitions (const X = () => or function X())
    comp_matches = re.findall(r'(?:const|function)\s+([A-Z][a-zA-Z0-9]*)\s*(?:=\s*\(|\()', ocr)
    components.extend(comp_matches)
    
    # Function definitions
    func_matches = re.findall(r'(?:const|function)\s+([a-z][a-zA-Z0-9]*)\s*(?:=\s*\(|\()', ocr)
    functions.extend(func_matches)

print(f'\nComponents defined: {Counter(components).most_common(20)}')
print(f'\nFunctions defined: {Counter(functions).most_common(20)}')

# Sample OCR
print('\n--- Sample OCR with code ---')
if code_chunks:
    print('Sample from first code chunk:')
    print(code_chunks[0].get('ocr_text', '')[:600])
