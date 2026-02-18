import json

with open('outputs/OPaLnMw2i_0_chunks_metadata.json') as f:
    data = json.load(f)

code_frames = []
for chunk in data.get('chunks', []):
    ocr = chunk.get('ocr_text', '')
    ocr_lower = ocr.lower()
    if 'timeline' in ocr_lower and ('import' in ocr_lower or 'const ' in ocr_lower or 'function' in ocr_lower or 'export' in ocr_lower or 'return' in ocr_lower):
        if len(ocr) > 300:
            code_frames.append({'ts': chunk.get('timestamp', 0), 'ocr': ocr})

print(f"Found {len(code_frames)} frames with substantial Timeline code\n")

for i, frame in enumerate(code_frames):
    print(f"=== Frame {i+1} at {frame['ts']:.1f}s ===")
    print(frame['ocr'])
    print("\n" + "="*60 + "\n")
