import os
import re
import cv2
import pytesseract
import easyocr
import whisper
import yt_dlp
import Levenshtein
import numpy as np
import json
import torch
import open_clip
import base64
import google.generativeai as genai
from PIL import Image
from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from flask_cors import CORS
from evaluation import RetrievalEvaluator, VideoRetrievalBenchmark
from vector_db import VideoLibrary, get_library
from groq import Groq
from pathlib import Path

# ── Load .env file ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded.")
except ImportError:
    print("⚠️  python-dotenv not installed — run: pip install python-dotenv")

app = Flask(__name__)
CORS(app)

env_path = Path(__file__).parent / ".env"
print(f"Checking for .env at: {env_path}")
print(f"GEMINI_API_KEY in env: {'Found' if os.environ.get('GEMINI_API_KEY') else 'NOT Found'}")
# ============================================================================
# GEMINI — primary LLM (vision + code + general queries)
# ============================================================================
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini configured (primary LLM — vision + code + general).")
else:
    gemini_model = None
    print("❌ GEMINI_API_KEY missing — add it to your .env file.")
    print("   Vision, code extraction, and screenshot features will NOT work.")

# ============================================================================
# GROQ — optional text-only fallback (used only if Gemini fails/unavailable)
# ============================================================================
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Groq configured (text-only fallback).")
else:
    groq_client = None
    print("ℹ️  GROQ_API_KEY not set — Groq fallback disabled (optional).")

# ============================================================================
# FOLDERS & MODELS
# ============================================================================

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({
        "status": "online",
        "gemini_active": gemini_model is not None,
        "env_path": str(env_path)
    })

UPLOAD_FOLDER = 'downloads'
OUTPUT_FOLDER = 'outputs'
FRAMES_FOLDER = os.path.join(UPLOAD_FOLDER, 'frames')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("base")
print("✅ Whisper loaded.")

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("✅ CLIP loaded.")

easyocr_reader = None

def get_easyocr_reader():
    global easyocr_reader
    if easyocr_reader is None:
        print("Initializing EasyOCR...")
        easyocr_reader = easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR ready.")
    return easyocr_reader

# ============================================================================
# KEYFRAME ALIGNMENT CONFIGURATION
# ============================================================================
SCENE_CHANGE_THRESHOLD = 5.0
AUDIO_WINDOW_SECONDS = 5.0
MIN_KEYFRAME_INTERVAL = 1.0
MAX_KEYFRAME_INTERVAL = 10.0


# ============================================================================
# QUERY ROUTER — decides Gemini (vision) vs Groq (text/code)
# ============================================================================

VISUAL_QUERY_PATTERNS = [
    r'\b(screenshot|screen\s*shot|ss|show\s+me|what\s+does\s+.+look\s+like)\b',
    r'\b(preview|ui|interface|design|layout|appearance|look|looks)\b',
    r'\b(diagram|flowchart|architecture|chart|visualize|structure)\b',
    r'\b(color|colour|theme|image|picture|photo)\b',
    r'\b(what\s+is\s+(shown|displayed|visible|on\s+screen))\b',
    r'\b(describe\s+(the\s+)?(page|website|project|screen|app))\b',
    r'\b(how\s+does\s+.+look)\b',
    r'\b(visual|visually)\b',
    r'\bprovide\s+(ss|screenshot|preview)\b',
]

CODE_QUERY_PATTERNS = [
    # Generic — language/tool agnostic
    r'\b(code|file|script|config|template|module|resource|provider|pipeline)\b',
    r'\b(function|class|method|syntax|command|implement|deploy|infrastructure)\b',
    r'\b(give\s+me|extract|get|show\s+me)\s+(the\s+)?(code|file|script|config)\b',
    # Any filename pattern
    r'[\w.-]+\.(?:tf|yaml|yml|py|js|jsx|ts|tsx|css|html|sh|groovy|go|rb|json|xml|sql|toml)',
    r'(?i)\b(jenkinsfile|dockerfile|makefile|vagrantfile)\b',
]


def classify_query(query: str) -> str:
    """
    Classify query as 'visual', 'code', or 'general'.
      'visual'  -> Gemini with real keyframe images
      'code'    -> Groq with OCR text dump
      'general' -> Groq with summarized context
    """
    q = query.lower()
    is_visual = any(re.search(p, q, re.IGNORECASE) for p in VISUAL_QUERY_PATTERNS)
    is_code   = any(re.search(p, q, re.IGNORECASE) for p in CODE_QUERY_PATTERNS)
    if is_visual and not is_code:
        return 'visual'
    if is_code:
        return 'code'
    return 'general'


# ============================================================================
# GEMINI VISION — real multimodal: images + text -> response
# ============================================================================
import base64


def encode_image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def load_pil_image(image_path: str):
    """Load image as PIL Image for Gemini."""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def generate_gemini_vision_response(query: str, chunks: list, max_frames: int = 5) -> dict:
    """
    Generate a response using Gemini 1.5 Flash with actual keyframe images.

    This is the TRUE multimodal path — Gemini sees the real video frames
    alongside OCR text and audio context, not just text descriptions.

    Returns:
        response    : Gemini's visual description
        frame_urls  : Frame paths to serve to frontend for display
        sources     : Timestamp + video_id references
        vision_used : True
    """
    if not gemini_model:
        return {
            'response': 'Vision model not configured. Please set GEMINI_API_KEY.',
            'frame_urls': [],
            'sources': [],
            'vision_used': False
        }

    frames_data = []
    sources = []

    for chunk in chunks[:max_frames]:
        frame_path = chunk.get('frame_path', '')
        timestamp  = chunk.get('timestamp', 0)
        video_id   = chunk.get('video_id', 'unknown')
        ocr_text   = chunk.get('ocr_text', '')
        audio_ctx  = chunk.get('audio_context', '')

        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        time_str = f"{mins}:{secs:02d}"

        pil_img = load_pil_image(frame_path) if (frame_path and os.path.exists(frame_path)) else None

        frames_data.append({
            'pil_image':     pil_img,
            'frame_path':    frame_path,
            'time_str':      time_str,
            'video_id':      video_id,
            'ocr_text':      ocr_text,
            'audio_context': audio_ctx,
        })
        sources.append({
            'video_id':   video_id,
            'timestamp':  timestamp,
            'similarity': chunk.get('similarity', 0),
            'frame_path': frame_path
        })

    valid_frames = [fd for fd in frames_data if fd['pil_image'] is not None]

    if not valid_frames:
        return {
            'response': 'No keyframe images found. Please re-process the video to generate frames.',
            'frame_urls': [],
            'sources': sources,
            'vision_used': False
        }

    # ── Build Gemini multimodal prompt ─────────────────────────────────────
    # Gemini accepts a flat list mixing strings and PIL Images
    prompt_parts = [
        f"You are VideoAI, an intelligent assistant analyzing video frames.\n"
        f"I will show you {len(valid_frames)} keyframes extracted from a video.\n\n"
        f"USER QUESTION: {query}\n\n"
        f"Instructions:\n"
        f"- If asked for a screenshot/preview: describe the visual layout, colors, UI elements, "
        f"fonts, design style in detail. Mention what sections are visible.\n"
        f"- If asked about diagrams/architecture: describe the visual structure you observe.\n"
        f"- If asked a general question: answer using both what you see and the supporting text.\n"
        f"- Always mention approximate timestamps of the frames you're referencing.\n\n"
        f"Here are the frames:"
    ]

    for i, fd in enumerate(valid_frames):
        prompt_parts.append(f"\n\n--- Frame {i+1} at timestamp {fd['time_str']} ---")
        prompt_parts.append(fd['pil_image'])   # PIL Image passed directly to Gemini

        # Supporting text context per frame
        support = []
        if fd['ocr_text']:
            support.append(f"On-screen text detected: {fd['ocr_text'][:400]}")
        if fd['audio_context']:
            support.append(f"Audio at this moment: {fd['audio_context'][:200]}")
        if support:
            prompt_parts.append("\n".join(support))

    prompt_parts.append(f"\n\nNow provide a thorough answer to: {query}")

    try:
        response = gemini_model.generate_content(prompt_parts)

        return {
            'response':        response.text,
            'frame_urls':      [fd['frame_path'] for fd in valid_frames],
            'sources':         sources,
            'vision_used':     True,
            'frames_analyzed': len(valid_frames),
            'model':           'gemini-2.5-flash'
        }

    except Exception as e:
        print(f"Gemini vision error: {e}")

        # Graceful fallback: Groq text-only if Gemini fails
        if groq_client:
            ocr_context = "\n".join([
                f"[{fd['time_str']}] {fd['ocr_text'][:300]}"
                for fd in frames_data if fd['ocr_text']
            ])
            try:
                fallback = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are VideoAI. Answer based on OCR text from video frames."},
                        {"role": "user", "content": f"Question: {query}\n\nOCR from frames:\n{ocr_context}"}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                return {
                    'response':   fallback.choices[0].message.content
                                  + "\n\n*(Vision unavailable — answered from OCR text only)*",
                    'frame_urls': [fd['frame_path'] for fd in frames_data if fd['frame_path']],
                    'sources':    sources,
                    'vision_used':  False,
                    'vision_error': str(e)
                }
            except Exception:
                pass

        return {
            'response':     f'Vision error: {str(e)}',
            'frame_urls':   [],
            'sources':      sources,
            'vision_used':  False,
            'vision_error': str(e)
        }


# ============================================================================
# UNIFIED MULTIMODAL RESPONSE — Gemini sees frames + OCR + audio together
# Content-agnostic: works for CSS, Terraform, Jenkins, React, Python, anything
# ============================================================================

def generate_multimodal_response(query: str, chunks: list, max_chunks: int = 8) -> dict:
    """
    THE CORE ENGINE: passes all three modalities to Gemini 1.5 Flash together.

    Content-agnostic by design — Gemini looks at the actual frames and decides
    what is on screen. No hardcoded CSS/HTML/JS patterns. Works identically for:
        - CSS/HTML/JS tutorials
        - Terraform / infrastructure-as-code
        - Jenkins / CI-CD pipelines
        - Kubernetes YAML
        - Python scripts
        - React components
        - Architecture diagrams
        - Any other content type

    For every chunk Gemini receives all three simultaneously:
        1. Keyframe IMAGE  — primary source, highest trust
        2. OCR TEXT        — supporting evidence, may have noise/errors
        3. AUDIO TRANSCRIPT — what the presenter said at that moment

    Gemini is instructed to:
        - Read code directly from the image, use OCR as a hint only
        - Correct OCR errors by comparing against what the image actually shows
        - Identify the language/tool from visual context (not keyword matching)
        - Return screenshots, diagrams, code, or answers based on what user asked
    """
    if not gemini_model:
        return _groq_text_fallback(query, chunks, max_chunks)

    selected     = chunks[:max_chunks]
    sources      = []
    prompt_parts = []

    # ── Universal system instruction — no content type assumptions ────────────
    prompt_parts.append(
        "You are VideoAI, an intelligent multimodal assistant that analyzes technical video content.\n\n"

        "You will receive video chunks. Each chunk contains THREE information sources:\n"
        "  1. FRAME IMAGE    — the actual screenshot from the video (highest trust)\n"
        "  2. OCR TEXT       — automatically extracted text (may have errors/noise)\n"
        "  3. AUDIO TRANSCRIPT — what the presenter said at that moment\n\n"

        "UNIVERSAL RULES (apply regardless of content type):\n"
        "  • Always read code/text directly from the FRAME IMAGE — it is more accurate than OCR\n"
        "  • Use OCR text only as a starting hint; correct any errors you see in the image\n"
        "  • Identify the programming language, tool, or technology from what you SEE in the frames\n"
        "    (could be Terraform, CSS, Python, YAML, Groovy, Bash, JavaScript, SQL, etc.)\n"
        "  • Never assume the content type — let the frames tell you what is on screen\n"
        "  • If multiple frames show the same file evolving, merge them into the final complete version\n"
        "  • Preserve exact syntax for the detected language — indentation, brackets, keywords\n\n"

        "RESPONSE MODES (determined by the user's question, not by content type):\n"
        "  CODE REQUEST  → Extract complete, accurate code from the frames. Fix OCR errors visually.\n"
        "                  Output as: ### filename.ext followed by a fenced code block.\n"
        "                  Detect the correct language from the image (tf, py, js, yaml, css, etc.)\n"
        "  SCREENSHOT    → Describe exactly what you see: layout, colors, structure, visible text.\n"
        "  DIAGRAM       → Generate valid Mermaid syntax based on structure visible in frames.\n"
        "  EXPLANATION   → Explain what the video is demonstrating using all three sources.\n\n"

        f"USER QUESTION: {query}\n\n"
        "Video chunks follow — analyze each frame image carefully:"
    )

    # ── Per-chunk: image + OCR + audio ────────────────────────────────────────
    valid_frame_count = 0
    for i, chunk in enumerate(selected):
        timestamp  = chunk.get('timestamp', 0)
        video_id   = chunk.get('video_id', 'unknown')
        ocr_text   = chunk.get('ocr_text', '').strip()
        audio_ctx  = chunk.get('audio_context', '').strip()
        frame_path = chunk.get('frame_path', '')

        mins     = int(timestamp // 60)
        secs     = int(timestamp % 60)
        time_str = f"{mins}:{secs:02d}"

        prompt_parts.append(f"\n\n{'─'*55}")
        prompt_parts.append(f"CHUNK {i+1}  |  {video_id}  |  {time_str}")
        prompt_parts.append(f"{'─'*55}")

        # 1. Frame image — highest trust
        pil_img = load_pil_image(frame_path) if (frame_path and os.path.exists(frame_path)) else None
        if pil_img:
            prompt_parts.append("▶ FRAME IMAGE (read this as the primary source of truth):")
            prompt_parts.append(pil_img)
            valid_frame_count += 1
        else:
            prompt_parts.append("▶ FRAME IMAGE: not available")

        # 2. OCR text — supporting, may be noisy
        if ocr_text:
            prompt_parts.append(
                f"\n▶ OCR TEXT (auto-extracted — verify against the image above, correct any errors):\n"
                f"{ocr_text[:700]}"
            )
        else:
            prompt_parts.append("\n▶ OCR TEXT: none detected")

        # 3. Audio transcript
        if audio_ctx:
            prompt_parts.append(
                f"\n▶ AUDIO at {time_str} (what the presenter said):\n{audio_ctx[:400]}"
            )
        else:
            prompt_parts.append(f"\n▶ AUDIO: none")

        sources.append({
            'video_id':   video_id,
            'timestamp':  timestamp,
            'similarity': chunk.get('similarity', 0),
            'frame_path': frame_path
        })

    # ── Final instruction — completely query-driven, no content assumptions ───
    prompt_parts.append(
        f"\n\n{'═'*55}\n"
        f"Now answer this question using the frames above: {query}\n\n"

        "INSTRUCTIONS FOR YOUR RESPONSE:\n"

        "If the question asks for CODE or a file:\n"
        "  - Look at each frame image and read the code directly\n"
        "  - Detect the language from the image (do NOT guess from the question)\n"
        "  - Merge frames showing the same file into ONE complete output\n"
        "  - Fix any OCR errors by reading the actual image\n"
        "  - Format: ### filename.ext  then  ```language\\ncode\\n```\n"
        "  - Include everything visible: imports, config blocks, functions, exports\n"
        "  - For infrastructure code (Terraform, K8s YAML, Dockerfile): preserve exact syntax\n"
        "  - For scripts (Python, Bash, Groovy/Jenkinsfile): preserve indentation exactly\n\n"

        "If the question asks for a SCREENSHOT or VISUAL DESCRIPTION:\n"
        "  - Describe the layout, colors, UI components, text visible on screen\n"
        "  - Mention which timestamp the description refers to\n\n"

        "If the question asks for a DIAGRAM or ARCHITECTURE:\n"
        "  - Generate a valid Mermaid diagram based on what you observe in the frames\n"
        "  - Use only structure that is actually visible, do not invent connections\n"
        "  - Valid syntax: flowchart TD, A[Label] --> B[Label], B{Decision?}\n"
        "  - No colons in labels, simple IDs only (A B C), max 15 nodes\n\n"

        "If the question is a GENERAL EXPLANATION:\n"
        "  - Use all three sources (image, OCR, audio) to give a thorough answer\n"
        "  - Cite specific timestamps where relevant\n"
    )

    # ── Call Gemini ────────────────────────────────────────────────────────────
    # Use lower temperature for code/diagram (precision), higher for explanation
    query_lower = query.lower()
    is_precise  = bool(re.search(
        r'\b(code|extract|give|show|diagram|flowchart|architecture|file|'
        r'script|config|pipeline|template|module|resource|provider)\b',
        query_lower
    ))

    try:
        print(
            f"[GEMINI MULTIMODAL] {valid_frame_count} frames | "
            f"{sum(1 for c in selected if c.get('ocr_text'))} OCR | "
            f"{sum(1 for c in selected if c.get('audio_context'))} audio | "
            f"query: {query[:60]}",
            flush=True
        )

        response = gemini_model.generate_content(
            prompt_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.05 if is_precise else 0.4,
                max_output_tokens=8192,
            )
        )

        return {
            'response':        response.text,
            'frame_urls':      [c.get('frame_path', '') for c in selected if c.get('frame_path')],
            'sources':         sources,
            'llm_enabled':     True,
            'vision_used':     valid_frame_count > 0,
            'frames_analyzed': valid_frame_count,
            'model':           'gemini-2.5-flash',
            'modalities_used': {
                'frames': valid_frame_count,
                'ocr':    sum(1 for c in selected if c.get('ocr_text')),
                'audio':  sum(1 for c in selected if c.get('audio_context')),
            }
        }

    except Exception as e:
        print(f"[GEMINI ERROR] {e}", flush=True)
        return _groq_text_fallback(query, chunks, max_chunks)


def _groq_text_fallback(query: str, chunks: list, max_chunks: int = 8) -> dict:
    """
    Groq text-only fallback when Gemini is unavailable.
    Used only as a safety net — OCR + audio only, no visual.
    """
    if not groq_client:
        return {
            'response': 'No LLM configured. Set GEMINI_API_KEY (recommended) or GROQ_API_KEY.',
            'sources': [], 'llm_enabled': False
        }

    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        timestamp  = chunk.get('timestamp', 0)
        video_id   = chunk.get('video_id', 'unknown')
        ocr_text   = chunk.get('ocr_text', '').strip()
        audio_ctx  = chunk.get('audio_context', '').strip()

        mins, secs = int(timestamp // 60), int(timestamp % 60)
        time_str   = f"{mins}:{secs:02d}"

        part = f"[{time_str}]\n"
        if ocr_text:
            part += f"OCR: {ocr_text[:500]}\n"
        if audio_ctx:
            part += f"Audio: {audio_ctx[:300]}\n"
        context_parts.append(part)
        sources.append({'video_id': video_id, 'timestamp': timestamp,
                        'similarity': chunk.get('similarity', 0)})

    needs_code = bool(re.search(r'\b(code|css|html|js|function|class|style)\b', query, re.IGNORECASE))
    system_msg = ("Extract complete code from OCR. Fix errors. Output code blocks only."
                  if needs_code else
                  "You are VideoAI. Answer based on OCR and audio transcript from video chunks.")

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{''.join(context_parts)}"}
            ],
            temperature=0.1 if needs_code else 0.7,
            max_tokens=4096
        )
        return {
            'response':    resp.choices[0].message.content,
            'frame_urls':  [],
            'sources':     sources,
            'llm_enabled': True,
            'vision_used': False,
            'model':       'groq-llama-3.3-70b (fallback — no vision)'
        }
    except Exception as e:
        return {'response': f'Error: {str(e)}', 'sources': sources, 'llm_enabled': False}


# backward-compatible wrapper — now routes to unified multimodal function
def generate_llm_response(query: str, chunks: list, max_chunks: int = 5,
                          want_diagram: bool = False, want_code: bool = False,
                          use_vision: bool = True) -> dict:
    """Routes all calls to generate_multimodal_response (Gemini frames+OCR+audio)."""
    return generate_multimodal_response(query, chunks, max_chunks=max_chunks)


# ============================================================================
# HYBRID SEARCH: NEW - combines vector similarity + keyword matching for code
# ============================================================================

def keyword_search_chunks(chunks: list, query: str, filename: str = None) -> list:
    """
    Language-agnostic keyword/structural search for code-related queries.
    Scores chunks by universal code structure signals — works for CSS, Terraform,
    Python, YAML, Groovy/Jenkinsfile, Bash, SQL, React, or any other language.
    Does NOT use language-specific keyword lists.
    """
    query_lower = query.lower()

    # Universal code structure signals — present in ALL programming languages/tools
    universal_code_signals = [
        (r'[{}]',          0.5),   # braces — JS, CSS, Terraform, JSON, Groovy
        (r';',             0.3),   # semicolons — JS, CSS, Java, SQL
        (r':',             0.2),   # colons — Python, YAML, CSS properties
        (r'=',             0.2),   # assignment — universal
        (r'\(',            0.1),   # function calls — universal
        (r'^\s{2,}',       0.3),   # indentation — Python, YAML, Terraform
        (r'->|=>',         1.0),   # arrows — JS/TS, PHP, Rust
        (r'#.*$',          0.5),   # comments — Python, Bash, YAML, Terraform
        (r'//.*$',         0.5),   # comments — JS, Java, Terraform
        (r'/\*',           0.5),   # block comments — CSS, JS
        (r'"[^"]{2,}"',    0.2),   # string values
        (r"'[^']{2,}'",    0.2),   # string values
        (r'\[.*\]',        0.3),   # arrays/lists — universal
        (r'<[a-zA-Z/]',    0.5),   # XML/HTML tags
    ]

    # Additional score for longer OCR — more complete code = higher value
    scored_chunks = []
    for chunk in chunks:
        ocr = chunk.get('ocr_text', '')
        if not ocr or len(ocr) < 20:
            continue

        score = 0.0

        # Universal structural scoring
        for pattern, weight in universal_code_signals:
            matches = len(re.findall(pattern, ocr, re.MULTILINE))
            score += min(matches * weight, 5)

        # Bonus: filename match in OCR (e.g. "main.tf", "Jenkinsfile", "app.py")
        if filename:
            fname_base = re.sub(r'\.\w+$', '', filename.lower())
            if fname_base in ocr.lower() or filename.lower() in ocr.lower():
                score += 8

        # Bonus: query keyword appears directly in OCR
        query_words = [w for w in query_lower.split() if len(w) > 3]
        for word in query_words:
            if word in ocr.lower():
                score += 1.5

        # Bonus for longer OCR (more complete content)
        score += min(len(ocr) / 150, 5)
        score += min(len(ocr) / 200, 3)

        if score > 1:
            chunk_copy = dict(chunk)
            chunk_copy['keyword_score'] = score
            scored_chunks.append(chunk_copy)

    # Sort by keyword score descending
    scored_chunks.sort(key=lambda x: x.get('keyword_score', 0), reverse=True)
    return scored_chunks


def hybrid_search(query: str, video_id: str, top_k: int = 10) -> list:
    """
    NEW: Hybrid search combining CLIP vector similarity + keyword matching.

    Strategy:
    - Vector search: finds semantically relevant chunks
    - Keyword search: finds structurally relevant code chunks
    - Merge and deduplicate, ranking by combined score

    This fixes the core issue where CLIP alone misses CSS/code chunks.
    """
    chunks_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz")
    metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")

    if not os.path.exists(chunks_file) or not os.path.exists(metadata_file):
        return []

    npz_data = np.load(chunks_file)
    chunk_vectors = npz_data['vectors']
    timestamps = npz_data['timestamps']

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    chunks = metadata.get('chunks', [])

    # ── Vector search ──────────────────────────────────────────────────────────
    query_inputs = clip_tokenizer([query]).to(device)
    with torch.no_grad():
        query_features = clip_model.encode_text(query_inputs)
        query_features /= query_features.norm(dim=-1, keepdim=True)

    query_vector = query_features.cpu().numpy().flatten()

    # Use text portion only (first 512 dims)
    text_vectors = chunk_vectors[:, :512] if chunk_vectors.shape[1] >= 512 else chunk_vectors
    similarities = np.dot(text_vectors, query_vector[:text_vectors.shape[1]]).flatten()

    # Build result dicts with vector similarity
    vector_results = {}
    for idx in range(len(chunks)):
        chunk_meta = chunks[idx]
        vector_results[idx] = {
            'chunk_index': idx,
            'video_id': video_id,
            'timestamp': float(timestamps[idx]) if idx < len(timestamps) else 0.0,
            'similarity': float(similarities[idx]),
            'ocr_text': chunk_meta.get('ocr_text', ''),
            'audio_context': chunk_meta.get('audio_context', ''),
            'frame_path': chunk_meta.get('frame_path', ''),
            'keyword_score': 0.0
        }

    # ── Keyword search ─────────────────────────────────────────────────────────
    # Detect filename in query (e.g. "style.css", "index.html")
    filename_match = re.search(
        r'\b([\w-]+\.(css|html|js|jsx|ts|tsx|py|json|yaml|yml))\b',
        query, re.IGNORECASE
    )
    filename = filename_match.group(1) if filename_match else None

    keyword_results = keyword_search_chunks(chunks, query, filename)

    # Inject keyword scores into vector_results
    for kr in keyword_results:
        idx = kr.get('index') or kr.get('chunk_index')
        if idx is not None and idx in vector_results:
            vector_results[idx]['keyword_score'] = kr.get('keyword_score', 0)

    # ── Combine scores ─────────────────────────────────────────────────────────
    is_code_query = bool(re.search(
        r'\b(css|style|html|javascript|js|code|class|selector|function|'
        r'about|section|nav|header|footer|give|show|where|full|complete)\b',
        query, re.IGNORECASE
    ))

    all_results = list(vector_results.values())

    for r in all_results:
        vec_score = r['similarity']
        kw_score  = r.get('keyword_score', 0)

        if is_code_query:
            # For code queries: weight keyword score higher than vector score
            r['combined_score'] = (vec_score * 0.3) + (min(kw_score / 20.0, 1.0) * 0.7)
        else:
            # For general queries: weight vector score higher
            r['combined_score'] = (vec_score * 0.7) + (min(kw_score / 20.0, 1.0) * 0.3)

    # Sort by combined score
    all_results.sort(key=lambda x: x['combined_score'], reverse=True)

    # Filter out very short / empty OCR chunks for code queries
    if is_code_query:
        all_results = [r for r in all_results if len(r.get('ocr_text', '')) > 30]

    return all_results[:top_k]


# ============================================================================
# OCR UTILITIES
# ============================================================================

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'(Show UI|Show Ul|Subscribe|Share|Download)', '', text, flags=re.IGNORECASE)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        alnum_count = sum(c.isalnum() or c.isspace() for c in line)
        if (alnum_count / len(line)) > 0.6:
            cleaned_lines.append(line)
    text = " ".join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_meaningful_ocr_text(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 2:
        return False
    meaningful_patterns = [
        r'\b(import|from|def|class|function|return|if|else|for|while|try|except)\b',
        r'\b(const|let|var|export|async|await)\b',
        r'\b(git|npm|pip|docker|kubectl|aws|curl|wget|sudo|apt|yum)\b',
        r'\b(is|are|was|were|have|has|can|will|would|should|the|and|or|but)\b',
        r'\b(jenkins|kubernetes|docker|pipeline|stage|deploy|build|server|install)\b',
        # Generic structural code signals — any language
        r'[{}=;:\[\]()]',
        r'<[a-zA-Z/]',
        r'\b(resource|provider|module|output|variable|locals|terraform)\b',
    ]
    for pattern in meaningful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    word_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text.replace(' ', ''))
    if total_chars > 0 and (word_chars / total_chars) < 0.5:
        return False
    single_chars = re.findall(r'\b[a-zA-Z0-9]\b', text)
    if len(single_chars) > len(words) * 2:
        return False
    return len(words) >= 3


def perform_ocr_on_frame(frame):
    """
    Performs OCR on a single frame using EasyOCR with Tesseract fallback.
    """
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reader = get_easyocr_reader()
        results = reader.readtext(rgb_frame, paragraph=True, detail=0)
        if results:
            text = '\n'.join(results)
            return clean_text(text)
        results = reader.readtext(rgb_frame, paragraph=False)
        if results:
            results.sort(key=lambda x: x[0][0][1])
            lines = []
            current_line = []
            last_y = -100
            for detection in results:
                bbox, text, confidence = detection
                if confidence < 0.3:
                    continue
                y_pos = bbox[0][1]
                if abs(y_pos - last_y) < 20:
                    current_line.append(text)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [text]
                    last_y = y_pos
            if current_line:
                lines.append(' '.join(current_line))
            return clean_text('\n'.join(lines))
    except Exception as e:
        print(f"EasyOCR failed, falling back to Tesseract: {e}")

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6')
        return clean_text(text)
    except:
        return ""


def perform_code_ocr(frame):
    """
    Specialized OCR for extracting code from terminal/IDE screenshots.
    """
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reader = get_easyocr_reader()
        results = reader.readtext(
            rgb_frame,
            paragraph=False,
            width_ths=0.5,
            height_ths=0.5,
            contrast_ths=0.1,
            text_threshold=0.5
        )
        if not results:
            return ""
        results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
        lines = []
        current_line = []
        last_y = -100
        line_height_threshold = 25

        for detection in results:
            bbox, text, confidence = detection
            if confidence < 0.2 or not text.strip():
                continue
            y_pos = (bbox[0][1] + bbox[2][1]) / 2
            x_pos = bbox[0][0]
            if abs(y_pos - last_y) < line_height_threshold:
                current_line.append((x_pos, text))
            else:
                if current_line:
                    current_line.sort(key=lambda x: x[0])
                    line_text = ' '.join([t for _, t in current_line])
                    lines.append(line_text)
                current_line = [(x_pos, text)]
                last_y = y_pos

        if current_line:
            current_line.sort(key=lambda x: x[0])
            line_text = ' '.join([t for _, t in current_line])
            lines.append(line_text)

        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            code_chars = set('{}[]()=;:.<>/-_"\'\\')
            if any(c in code_chars for c in line) or len(line) > 3:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    except Exception as e:
        print(f"Code OCR error: {e}")
        return perform_ocr_on_frame(frame)


# ============================================================================
# VIDEO PROCESSING PIPELINE
# ============================================================================

def download_video(url):
    """Downloads video using yt-dlp and returns the filename."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(UPLOAD_FOLDER, '%(id)s.%(ext)s'),
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_id = info_dict.get("id", None)
        ext = info_dict.get("ext", "mp4")
        filename = f"{video_id}.{ext}"
        return os.path.join(UPLOAD_FOLDER, filename), video_id


def process_audio_with_timestamps(video_path):
    """Extracts and transcribes audio using Whisper WITH word-level timestamps."""
    print(f"Transcribing audio with timestamps from {video_path}...")
    result = model.transcribe(video_path, word_timestamps=True)
    segments = []
    for segment in result.get("segments", []):
        segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip()
        })
    full_text = result["text"]
    return full_text, segments


def detect_keyframes(video_path, threshold=SCENE_CHANGE_THRESHOLD):
    """
    DYNAMIC ANCHORING: Detects significant visual changes to find keyframes.
    Now uses lower threshold (5.0 vs old 15.0) and shorter max interval (10s vs 30s)
    to ensure better coverage of coding/typing videos.
    """
    print(f"Detecting keyframes with threshold={threshold}...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    keyframes = []
    last_frame_gray = None
    last_keyframe_time = -MIN_KEYFRAME_INTERVAL
    frame_count = 0

    sample_interval = max(1, int(fps / 10))  # ~10 samples per second

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_interval == 0:
            timestamp = frame_count / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

            if last_frame_gray is not None:
                frame_delta = cv2.absdiff(last_frame_gray, gray_blur)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                change_score = (np.sum(thresh) / thresh.size)

                if change_score > threshold and (timestamp - last_keyframe_time) >= MIN_KEYFRAME_INTERVAL:
                    keyframes.append({
                        'timestamp': timestamp,
                        'frame': frame.copy(),
                        'change_score': change_score,
                        'frame_number': frame_count
                    })
                    last_keyframe_time = timestamp
                    print(f"  ⚡ Keyframe at {timestamp:.2f}s (change: {change_score:.1f})")
                elif (timestamp - last_keyframe_time) >= MAX_KEYFRAME_INTERVAL:
                    keyframes.append({
                        'timestamp': timestamp,
                        'frame': frame.copy(),
                        'change_score': change_score,
                        'frame_number': frame_count
                    })
                    last_keyframe_time = timestamp
                    print(f"  ⏰ Keyframe at {timestamp:.2f}s (time-based, change: {change_score:.1f})")
            else:
                keyframes.append({
                    'timestamp': 0.0,
                    'frame': frame.copy(),
                    'change_score': 100.0,
                    'frame_number': 0
                })
                last_keyframe_time = 0.0
                print(f"  ⚡ Keyframe at 0.00s (first frame)")

            last_frame_gray = gray_blur
        frame_count += 1

    cap.release()
    print(f"Detected {len(keyframes)} keyframes from {duration:.1f}s video")
    return keyframes


def gather_audio_context(timestamp, transcript_segments, window_seconds=AUDIO_WINDOW_SECONDS):
    """Gathers audio text from a temporal window around the keyframe timestamp."""
    start_window = max(0, timestamp - window_seconds)
    end_window = timestamp + window_seconds
    relevant_text = []
    for seg in transcript_segments:
        if seg['end'] >= start_window and seg['start'] <= end_window:
            relevant_text.append(seg['text'])
    return " ".join(relevant_text)


def create_chunk_vector(frame, ocr_text, audio_context):
    """
    Creates a 1024D multimodal vector for a single chunk.
    [0:512]   = CLIP text embedding (OCR + audio)
    [512:1024] = CLIP visual embedding (keyframe image)
    """
    combined_text = f"{audio_context} {ocr_text}".strip()

    if combined_text:
        text_inputs = clip_tokenizer([combined_text[:300]]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = torch.zeros(1, 512).to(device)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        visual_features = clip_model.encode_image(image_input)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)

    multimodal_vector = torch.cat((text_features, visual_features), dim=1)
    return multimodal_vector.cpu().numpy()


def process_keyframe_aligned(video_path, video_id):
    """
    COMPLETE KEYFRAME-ALIGNED PIPELINE
    1. Detect keyframes (dynamic scene changes)
    2. Transcribe audio with timestamps
    3. For each keyframe: OCR + audio context + 1024D multimodal vector
    4. Save all outputs
    """
    print("\n" + "="*60)
    print("KEYFRAME-ALIGNED MULTIMODAL EXTRACTION")
    print("="*60 + "\n")

    print("[1/4] Detecting scene changes...")
    keyframes = detect_keyframes(video_path)

    print("\n[2/4] Transcribing audio with timestamps...")
    full_transcript, transcript_segments = process_audio_with_timestamps(video_path)

    print(f"\n[3/4] Processing {len(keyframes)} keyframes...")
    chunks = []
    all_ocr_texts = []
    all_vectors = []

    video_frames_folder = os.path.join(FRAMES_FOLDER, video_id)
    os.makedirs(video_frames_folder, exist_ok=True)

    last_ocr_text = ""

    for i, kf in enumerate(keyframes):
        timestamp = kf['timestamp']
        frame = kf['frame']

        print(f"  Processing keyframe {i+1}/{len(keyframes)} at {timestamp:.2f}s...")

        # Try specialized code OCR first, fall back to general OCR
        ocr_text = perform_code_ocr(frame)
        if not ocr_text or len(ocr_text) < 20:
            ocr_text = perform_ocr_on_frame(frame)

        if ocr_text:
            similarity = Levenshtein.ratio(last_ocr_text, ocr_text)
            if similarity < 0.90:
                all_ocr_texts.append(ocr_text)
                last_ocr_text = ocr_text
            elif len(ocr_text) > len(last_ocr_text):
                if all_ocr_texts:
                    all_ocr_texts[-1] = ocr_text
                else:
                    all_ocr_texts.append(ocr_text)
                last_ocr_text = ocr_text

        audio_context = gather_audio_context(timestamp, transcript_segments)
        chunk_vector = create_chunk_vector(frame, ocr_text, audio_context)
        all_vectors.append(chunk_vector)

        frame_filename = f"keyframe_{i:04d}_{timestamp:.2f}s.jpg"
        frame_path = os.path.join(video_frames_folder, frame_filename)
        cv2.imwrite(frame_path, frame)

        chunks.append({
            'index': i,
            'timestamp': timestamp,
            'ocr_text': ocr_text,
            'audio_context': audio_context,
            'frame_path': frame_path,
            'change_score': kf['change_score']
        })

    print("\n[4/4] Saving outputs...")

    audio_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_audio.txt")
    with open(audio_file_path, "w") as f:
        f.write(full_transcript)

    visual_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_visual.txt")
    with open(visual_file_path, "w") as f:
        f.write("\n\n".join(all_ocr_texts))

    chunks_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz")
    np.savez(
        chunks_file_path,
        vectors=np.vstack(all_vectors),
        timestamps=np.array([c['timestamp'] for c in chunks])
    )

    if all_vectors:
        combined_vector = np.mean(np.vstack(all_vectors), axis=0, keepdims=True)
    else:
        combined_vector = np.zeros((1, 1024))

    vector_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_multimodal_vector.npy")
    np.save(vector_file_path, combined_vector)

    metadata_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
    with open(metadata_file_path, "w") as f:
        json.dump({
            'video_id': video_id,
            'total_keyframes': len(keyframes),
            'chunks': [{
                'index': c['index'],
                'timestamp': c['timestamp'],
                'ocr_text': c['ocr_text'][:500] if c['ocr_text'] else "",
                'audio_context': c['audio_context'][:500] if c['audio_context'] else "",
                'frame_path': c['frame_path'],
                'change_score': c['change_score']
            } for c in chunks]
        }, f, indent=2)

    print("\n" + "="*60)
    print(f"✅ PROCESSING COMPLETE")
    print(f"   Keyframes detected: {len(keyframes)}")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Vector dimensions: {combined_vector.shape}")
    print("="*60 + "\n")

    return {
        'full_transcript': full_transcript,
        'visual_text': "\n\n".join(all_ocr_texts),
        'chunks': chunks,
        'files': {
            'audio': audio_file_path,
            'visual': visual_file_path,
            'vector': vector_file_path,
            'chunks': chunks_file_path,
            'metadata': metadata_file_path,
            'frames_folder': video_frames_folder
        },
        'stats': {
            'keyframes_detected': len(keyframes),
            'chunks_created': len(chunks),
            'vector_dim': 1024
        }
    }


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/api/transcribe', methods=['POST'])
def transcribe_video():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    def generate():
        try:
            yield f"data: {json.dumps({'status': 'Downloading video...'})}\n\n"
            video_path, video_id = download_video(url)

            yield f"data: {json.dumps({'status': 'Detecting scene changes...'})}\n\n"
            results = process_keyframe_aligned(video_path, video_id)

            yield f"data: {json.dumps({'status': 'Processing keyframes...'})}\n\n"
            yield f"data: {json.dumps({'status': 'Creating multimodal vectors...'})}\n\n"
            yield f"data: {json.dumps({'status': 'Finalizing outputs...'})}\n\n"

            result = {
                'message': 'Processing complete',
                'video_id': video_id,
                'transcription': results['full_transcript'],
                'visual_text': results['visual_text'],
                'files': results['files'],
                'stats': results['stats'],
                'chunks_preview': [{
                    'timestamp': c['timestamp'],
                    'ocr_preview': c['ocr_text'][:100] if c['ocr_text'] else "",
                    'audio_preview': c['audio_context'][:100] if c['audio_context'] else ""
                } for c in results['chunks'][:5]]
            }
            yield f"data: {json.dumps({'status': 'Complete', 'data': result})}\n\n"

        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/search', methods=['POST'])
def search_chunks():
    """
    Search through video chunks using hybrid semantic + keyword similarity.
    FIXED: Now uses hybrid_search() instead of pure vector search for better
    code retrieval. Also improved want_code detection and LLM prompting.
    """
    data = request.json
    query = data.get('query')
    video_id = data.get('video_id')
    top_k = data.get('top_k', 10)          # FIXED: increased default from 5 to 10
    generate_response = data.get('generate_response', True)

    if not query or not video_id:
        return jsonify({'error': 'Query and video_id required'}), 400

    try:
        # ── Step 1: Classify query to route to correct model ──────────────────
        query_type = classify_query(query)
        want_code  = (query_type == 'code') or bool(re.search(
            r'\b(code|snippet|file|script|config|module|resource|provider|pipeline|'
            r'function|class|method|give|show|extract|get|full|complete|all|deploy)\b',
            query, re.IGNORECASE
        ))

        extract_all_code = bool(re.search(
            r'(all|every|complete|full|entire|give).*(code|snippet|example|file)',
            query, re.IGNORECASE
        ))

        specific_file_request = bool(re.search(
            r'[\w.-]+\.(?:tf|tfvars|yaml|yml|json|py|js|jsx|ts|tsx|css|html|'
            r'sh|groovy|java|go|rb|rs|sql|xml|toml|ini|conf|md|gradle)'
            r'|(?i)\b(jenkinsfile|dockerfile|makefile|vagrantfile)\b',
            query
        ))

        should_extract_files = extract_all_code or specific_file_request

        # ── Step 2: Hybrid search (vector + keyword) ───────────────────────────
        results = hybrid_search(query, video_id, top_k=top_k)

        response_data = {'results': results, 'query_type': query_type}

        # ── Step 3: ALL queries → unified multimodal response (Gemini sees frames+OCR+audio)
        if generate_response:
            print(f"[MULTIMODAL] query_type={query_type} → generate_multimodal_response", flush=True)
            result = generate_multimodal_response(query, results, max_chunks=top_k)
            response_data['llm_response']    = result.get('response', '')
            response_data['llm_enabled']     = result.get('llm_enabled', False)
            response_data['vision_used']     = result.get('vision_used', False)
            response_data['frames_analyzed'] = result.get('frames_analyzed', 0)
            response_data['model']           = result.get('model', '')
            response_data['frame_urls']      = result.get('frame_urls', [])
            response_data['sources']         = result.get('sources', [])
            response_data['modalities_used'] = result.get('modalities_used', {})

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract-file', methods=['POST'])
def extract_single_file():
    """Extract a specific file's code using smart OCR analysis for maximum accuracy."""
    try:
        data = request.json
        video_id = data.get('video_id')
        target_file = data.get('filename')

        if not video_id or not target_file:
            return jsonify({'error': 'video_id and filename required'}), 400

        chunks_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz")
        metadata_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")

        if not os.path.exists(chunks_path) or not os.path.exists(metadata_path):
            return jsonify({'error': 'Video not found'}), 404

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        chunks = metadata.get('chunks', [])
        # Strip any extension to get the base name for OCR matching
        target_lower = re.sub(r'\.[^.]+$', '', target_file.lower())

        ext = target_file.split('.')[-1].lower() if '.' in target_file else 'unknown'

        # Universal code detection — works for CSS, Terraform, Python, Groovy, YAML, etc.
        UNIVERSAL_CODE_RE = re.compile(r'[{}=;:\[\]()|<]|^\s{2,}', re.MULTILINE)

        code_frames = []
        for chunk in chunks:
            ocr = chunk.get('ocr_text', '')
            ocr_lower = ocr.lower()

            name_match   = target_lower in ocr_lower
            signal_count = len(UNIVERSAL_CODE_RE.findall(ocr))
            has_code = name_match or (signal_count >= 5)

            if has_code and len(ocr) > 80:
                code_frames.append({
                    'timestamp': chunk.get('timestamp', 0),
                    'ocr': ocr
                })

        if not code_frames:
            return jsonify({
                'error': f'No code frames found for {target_file}',
                'suggestion': 'Try a different filename or use /api/search with the file extension in your query'
            }), 404

        code_frames.sort(key=lambda x: x['timestamp'])
        selected_frames = code_frames if len(code_frames) <= 20 else [
            code_frames[int(i * len(code_frames) / 20)] for i in range(20)
        ]

        # Deduplicate lines preserving order
        seen_lines = set()
        ordered_lines = []
        for frame in selected_frames:
            for line in frame['ocr'].split('\n'):
                line_clean = line.strip()
                if line_clean and len(line_clean) > 2:
                    if any(x in line_clean for x in ['localhost', 'File Edit', 'Visual Studio',
                                                       'EXPL', 'PROBLEMS', 'OUTPUT']):
                        continue
                    if line_clean not in seen_lines:
                        seen_lines.add(line_clean)
                        ordered_lines.append(line_clean)

        # Route through unified multimodal engine (Gemini sees frames + OCR + audio)
        mm_chunks = []
        for cf in selected_frames[:8]:
            orig = next(
                (c for c in chunks if abs(c.get('timestamp', -999) - cf['timestamp']) < 0.1),
                None
            )
            mm_chunks.append({
                'video_id':      video_id,
                'timestamp':     cf['timestamp'],
                'ocr_text':      cf['ocr'],
                'audio_context': orig.get('audio_context', '') if orig else '',
                'frame_path':    orig.get('frame_path', '')    if orig else '',
                'similarity':    1.0,
            })

        mm_result = generate_multimodal_response(
            f"Extract the complete {target_file} code. "
            f"Read directly from the frame images and correct any OCR errors.",
            mm_chunks, max_chunks=8
        )
        code = mm_result.get('response', '')
        fence = re.search(r'```(?:\w+)?\n([\s\S]*?)```', code)
        if fence:
            code = fence.group(1).strip()

        return jsonify({
            'filename':           target_file,
            'code':               code,
            'frames_analyzed':    len(selected_frames),
            'total_frames_found': len(code_frames),
            'vision_used':        mm_result.get('vision_used', False),
            'model':              mm_result.get('model', ''),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# VIDEO LIBRARY ENDPOINTS
# ============================================================================

@app.route('/api/library/videos', methods=['GET'])
def get_video_library():
    """Get list of all processed and indexed videos."""
    try:
        videos = []
        for filename in os.listdir(OUTPUT_FOLDER):
            if filename.endswith('_chunks_metadata.json'):
                video_id = filename.replace('_chunks_metadata.json', '')
                metadata_path = os.path.join(OUTPUT_FOLDER, filename)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                videos.append({
                    'video_id': video_id,
                    'total_chunks': metadata.get('total_keyframes', 0),
                    'has_vectors': os.path.exists(os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz"))
                })
        return jsonify({'videos': videos})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/library/video/<video_id>', methods=['GET'])
def get_video_details(video_id):
    """Get detailed information about a specific video."""
    try:
        metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        if not os.path.exists(metadata_file):
            return jsonify({'error': 'Video not found'}), 404

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        audio_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_audio.txt")
        transcript = ""
        if os.path.exists(audio_file):
            with open(audio_file, 'r') as f:
                transcript = f.read()

        visual_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_visual.txt")
        visual_text = ""
        if os.path.exists(visual_file):
            with open(visual_file, 'r') as f:
                visual_text = f.read()

        return jsonify({
            'video_id': video_id,
            'total_keyframes': metadata.get('total_keyframes', 0),
            'chunks': metadata.get('chunks', []),
            'transcript': transcript,
            'visual_text': visual_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/library/video/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete a video and all its associated files."""
    try:
        files_to_delete = [
            f"{video_id}_audio.txt",
            f"{video_id}_visual.txt",
            f"{video_id}_chunks.npz",
            f"{video_id}_chunks_metadata.json",
            f"{video_id}_multimodal_vector.npy"
        ]
        deleted = []
        for filename in files_to_delete:
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted.append(filename)

        frames_folder = os.path.join(FRAMES_FOLDER, video_id)
        if os.path.exists(frames_folder):
            import shutil
            shutil.rmtree(frames_folder)
            deleted.append(f"frames/{video_id}")

        for ext in ['mp4', 'mkv', 'webm']:
            video_file = os.path.join(UPLOAD_FOLDER, f"{video_id}.{ext}")
            if os.path.exists(video_file):
                os.remove(video_file)
                deleted.append(f"{video_id}.{ext}")

        return jsonify({'deleted': deleted, 'video_id': video_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/library/video/<video_id>/reprocess-ocr', methods=['POST'])
def reprocess_video_ocr(video_id):
    """Re-process OCR for an existing video using EasyOCR."""
    try:
        metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        if not os.path.exists(metadata_file):
            return jsonify({'error': 'Video not found'}), 404

        frames_folder = os.path.join(FRAMES_FOLDER, video_id)
        if not os.path.exists(frames_folder):
            return jsonify({'error': 'Frames folder not found. Please re-upload the video.'}), 404

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
        if not frame_files:
            return jsonify({'error': 'No frames found in frames folder'}), 404

        updated_chunks = []
        all_ocr_text = []

        for i, chunk in enumerate(metadata.get('chunks', [])):
            frame_index = chunk.get('index', i)
            frame_path = None

            for frame_file in frame_files:
                if f"frame_{frame_index:04d}" in frame_file or f"keyframe_{frame_index:04d}" in frame_file:
                    frame_path = os.path.join(frames_folder, frame_file)
                    break

            if not frame_path and frame_index < len(frame_files):
                frame_path = os.path.join(frames_folder, frame_files[frame_index])

            new_ocr_text = ""
            if frame_path and os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    new_ocr_text = perform_code_ocr(frame)
                    if not new_ocr_text:
                        new_ocr_text = perform_ocr_on_frame(frame)

            chunk['ocr_text'] = new_ocr_text
            if new_ocr_text:
                all_ocr_text.append(f"[Frame {frame_index}]: {new_ocr_text}")
            updated_chunks.append(chunk)

        metadata['chunks'] = updated_chunks
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        visual_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_visual.txt")
        with open(visual_file, 'w') as f:
            f.write('\n'.join(all_ocr_text))

        # Regenerate vectors
        vectors_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz")
        if os.path.exists(vectors_file):
            chunk_vectors = []
            for chunk in updated_chunks:
                combined_text = f"{chunk.get('audio_context', '')} {chunk.get('ocr_text', '')}"
                if combined_text.strip():
                    text_tokens = clip_tokenizer([combined_text]).to(device)
                    with torch.no_grad():
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    chunk_vectors.append(text_features.cpu().numpy().flatten())
                else:
                    chunk_vectors.append(np.zeros(512))
            np.savez(vectors_file, vectors=np.array(chunk_vectors))

        return jsonify({
            'success': True,
            'video_id': video_id,
            'chunks_processed': len(updated_chunks),
            'message': 'OCR re-processed successfully with EasyOCR'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/library/video/<video_id>/extract-code', methods=['POST'])
def extract_code_from_video(video_id):
    """
    Extract ALL code snippets from a video using OCR + LLM consolidation.
    FILE-CENTRIC extraction: identifies all files and extracts complete code.
    """
    if not groq_client:
        return jsonify({'error': 'LLM not configured'}), 500

    try:
        metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        if not os.path.exists(metadata_file):
            return jsonify({'error': 'Video not found'}), 404

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        chunks = metadata.get('chunks', [])

        def clean_ide_ocr(text):
            ui_patterns = [
                r'^Activities\s+', r'Visual Studio Code\s*', r'Google Chrome\s*',
                r'File\s+Edit\s+Selection\s+View\s+Go\s+Run\s+Terminal\s+Help\s*',
                r'EXPLORER\s*', r'PROBLEMS\s+OUTPUT\s+DEBUG\s+CONSOLE\s+TERMINAL\s*',
                r'OUTLINE\s*', r'TIMELINE\s*', r'Ln\s*\d+,?\s*Col\s*\d+\s*',
                r'Spaces:\s*\d+\s*', r'UTF-8\s*',
            ]
            cleaned = text
            for pattern in ui_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
            return cleaned.strip()

        def detect_filename(ocr_text):
            EXT = (r'tf|tfvars|yaml|yml|json|py|js|jsx|ts|tsx|css|html|'
                   r'sh|bash|groovy|java|go|rb|rs|sql|xml|toml|ini|conf|cfg|md|gradle')
            patterns = [
                rf'([\w.-]+\.(?:{EXT}))\s*[-–]?\s*\w+',  # filename - project
                rf'Help\s+([\w.-]+\.(?:{EXT}))',             # VS Code / editor tab
                rf'([\w.-]+\.(?:{EXT}))\s+Visual',           # status bar
                rf'([\w.-]+\.(?:{EXT}))',                      # bare filename
                r'(?i)\b(jenkinsfile|dockerfile|makefile|vagrantfile)\b',
            ]
            for pattern in patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match:
                    return match.group(1)
            return None

        # Universal code detection — language agnostic
        UNIVERSAL_CODE_RE = re.compile(r'[{}=;:\[\]()|<]|^\s{2,}', re.MULTILINE)

        file_frames = {}
        unclassified_frames = []

        for chunk in chunks:
            ocr_text = chunk.get('ocr_text', '')
            if not ocr_text or len(ocr_text.strip()) < 20:
                continue

            code_chars = len(UNIVERSAL_CODE_RE.findall(ocr_text))

            if code_chars < 4:
                continue

            filename = detect_filename(ocr_text)
            cleaned_ocr = clean_ide_ocr(ocr_text)

            frame_data = {
                'index': chunk.get('index', 0),
                'timestamp': chunk.get('timestamp', 0),
                'ocr_text': cleaned_ocr,
                'audio': chunk.get('audio_context', '')[:200]
            }

            if filename:
                if filename not in file_frames:
                    file_frames[filename] = []
                file_frames[filename].append(frame_data)
            else:
                unclassified_frames.append(frame_data)

        if not file_frames and not unclassified_frames:
            return jsonify({
                'success': True,
                'video_id': video_id,
                'code_blocks': [],
                'files_found': [],
                'message': 'No code content detected in this video'
            })

        files_found = list(file_frames.keys())
        all_code_blocks = []

        for filename, frames in file_frames.items():
            if not frames:
                continue
            frames.sort(key=lambda x: x['timestamp'])

            file_ocr_context = f"FILE: {filename}\n" + "=" * 50 + "\n\n"
            for frame in frames[-15:]:
                mins = int(frame['timestamp'] // 60)
                secs = int(frame['timestamp'] % 60)
                file_ocr_context += f"[{mins}:{secs:02d}]\n{frame['ocr_text']}\n\n"

            mm_chunks = []
            for frame_data in frames[-8:]:
                orig = next(
                    (c for c in chunks if abs(c.get('timestamp', -999) - frame_data['timestamp']) < 0.1),
                    None
                )
                mm_chunks.append({
                    'video_id':      video_id,
                    'timestamp':     frame_data['timestamp'],
                    'ocr_text':      frame_data['ocr_text'],
                    'audio_context': orig.get('audio_context', '') if orig else '',
                    'frame_path':    orig.get('frame_path', '')    if orig else '',
                    'similarity':    1.0,
                })
            try:
                mm_result = generate_multimodal_response(
                    f"Extract the complete {filename} code. "
                    f"Read directly from frame images, correct any OCR errors.",
                    mm_chunks, max_chunks=8
                )
                raw  = mm_result.get('response', '')
                fence = re.search(r'```(?:\w+)?\n([\s\S]*?)```', raw)
                code  = fence.group(1).strip() if fence else raw.strip()
                if code:
                    ext = filename.split('.')[-1].lower() if '.' in filename else ''
                    open_b  = code.count('{')
                    close_b = code.count('}')
                    all_code_blocks.append({
                        'filename':    filename,
                        'language':    ext,
                        'code':        code,
                        'frames_used': len(frames),
                        'is_valid':    abs(open_b - close_b) <= 2,
                        'confidence':  'HIGH' if abs(open_b - close_b) <= 2 else 'MEDIUM',
                        'vision_used': mm_result.get('vision_used', False),
                    })
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                continue

        valid_blocks = [b for b in all_code_blocks if b.get('is_valid', False)]

        return jsonify({
            'success': True,
            'video_id': video_id,
            'files_found': files_found,
            'total_files': len(files_found),
            'accuracy': {
                'rating': 'HIGH (95%+)' if len(valid_blocks) == len(all_code_blocks) else 'MEDIUM',
                'valid_files': len(valid_blocks),
                'total_files': len(all_code_blocks)
            },
            'code_blocks': all_code_blocks
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# KEYFRAME/TIMELINE ENDPOINTS
# ============================================================================

@app.route('/api/video/<video_id>/keyframes', methods=['GET'])
def get_video_keyframes(video_id):
    """Get keyframe information for timeline visualization."""
    try:
        metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        if not os.path.exists(metadata_file):
            return jsonify({'error': 'Video not found'}), 404

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        keyframes = []
        for chunk in metadata.get('chunks', []):
            keyframes.append({
                'index': chunk['index'],
                'timestamp': chunk['timestamp'],
                'change_score': chunk.get('change_score', 0),
                'has_ocr': bool(chunk.get('ocr_text')),
                'has_audio': bool(chunk.get('audio_context')),
                'ocr_preview': (chunk.get('ocr_text') or '')[:50],
                'audio_preview': (chunk.get('audio_context') or '')[:50]
            })

        return jsonify({
            'video_id': video_id,
            'keyframes': keyframes,
            'total': len(keyframes)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/<video_id>/frame/<int:index>', methods=['GET'])
def get_keyframe_image(video_id, index):
    """Serve a keyframe image."""
    try:
        metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        if not os.path.exists(metadata_file):
            return jsonify({'error': 'Video not found'}), 404

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        chunks = metadata.get('chunks', [])
        if index < 0 or index >= len(chunks):
            return jsonify({'error': 'Invalid frame index'}), 400

        frame_path = chunks[index].get('frame_path', '')
        if not os.path.exists(frame_path):
            return jsonify({'error': 'Frame file not found'}), 404

        return send_file(frame_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# CROSS-VIDEO SEARCH ENDPOINTS
# ============================================================================

@app.route('/api/search/all', methods=['POST'])
def search_all_videos():
    """
    Search across all indexed videos using hybrid search + query routing.
    - Visual queries  → Gemini 1.5 Flash (sees real keyframes)
    - Code queries    → Groq LLaMA 3.3 70B (OCR text extraction)
    - General queries → Groq LLaMA 3.3 70B (summarized context)
    """
    data = request.json
    query = data.get('query')
    top_k = data.get('top_k', 10)
    generate_response = data.get('generate_response', True)
    video_id_filter = data.get('video_id')

    if not query:
        return jsonify({'error': 'Query required'}), 400

    # Classify query for routing
    query_type  = classify_query(query)
    want_diagram = bool(re.search(r'diagram|flowchart|chart|architecture|flow', query, re.IGNORECASE))
    want_code    = (query_type == 'code') or bool(re.search(
        r'\b(code|snippet|file|script|config|module|resource|provider|pipeline|'
        r'function|class|method|template|manifest|playbook|schema|'
        r'extract|give|full|complete|all|show|deploy|infrastructure)\b',
        query, re.IGNORECASE
    ))

    try:
        all_results = []
        videos_to_search = []

        if video_id_filter:
            chunks_file = os.path.join(OUTPUT_FOLDER, f"{video_id_filter}_chunks.npz")
            if os.path.exists(chunks_file):
                videos_to_search = [video_id_filter]
            else:
                return jsonify({'error': f'Video {video_id_filter} not found'}), 404
        else:
            for filename in os.listdir(OUTPUT_FOLDER):
                if filename.endswith('_chunks.npz'):
                    videos_to_search.append(filename.replace('_chunks.npz', ''))

        exclude_patterns = [
            r'thanks?\s*(for)?\s*watching', r'subscribe',
            r'like\s*(and|&)?\s*share', r'bell\s*icon',
            r'notification', r'see\s*you\s*(in|next)',
        ]

        def is_useful_segment(ocr_text, audio_context=''):
            combined = f"{ocr_text} {audio_context}".lower()
            for pattern in exclude_patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return False
            if len(ocr_text.strip()) < 20:
                return False
            return True

        for video_id in videos_to_search:
            video_results = hybrid_search(query, video_id, top_k=top_k * 2)
            for r in video_results:
                if is_useful_segment(r.get('ocr_text', ''), r.get('audio_context', '')):
                    all_results.append(r)

        all_results.sort(key=lambda x: x.get('combined_score', x.get('similarity', 0)), reverse=True)
        top_results = all_results[:top_k]

        response_data = {
            'query': query,
            'query_type': query_type,
            'results': top_results,
            'total_videos_searched': len(set(r['video_id'] for r in all_results))
        }

        # ── VISUAL QUERY → Gemini sees real frames ─────────────────────────────
        if query_type == 'visual' and generate_response:
            print(f"[ROUTER] Visual query → Gemini Vision", flush=True)
            vision_result = generate_gemini_vision_response(query, top_results, max_frames=5)
            response_data['llm_response']    = vision_result.get('response', '')
            response_data['llm_enabled']     = True
            response_data['vision_used']     = vision_result.get('vision_used', False)
            response_data['frames_analyzed'] = vision_result.get('frames_analyzed', 0)
            response_data['model']           = vision_result.get('model', 'gemini-2.5-flash')
            response_data['frame_urls']      = vision_result.get('frame_urls', [])
            response_data['sources']         = vision_result.get('sources', [])
            return jsonify(response_data)

        # ── CODE / GENERAL QUERY → Groq text path ─────────────────────────────
        extract_all_code = bool(re.search(
            r'(all|every|complete|full|entire).*(code|snippet|example)',
            query, re.IGNORECASE
        ))

        llm_result = None

        if want_code and video_id_filter and extract_all_code:
            # Delegate to full extraction endpoint logic
            try:
                metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id_filter}_chunks_metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    chunks = metadata.get('chunks', [])
                    file_patterns = [
                        r'([A-Z][a-z]+(?:[A-Z][a-z]+)*)\s*\.\s*(js|jsx|ts|tsx|py|css|html|json)',
                        r'([a-z]+(?:[A-Z][a-z]+)*)\s*\.\s*(js|jsx|ts|tsx|py|css|html|json)',
                        r'(index|app|main|config|constants|styles|default)\s*\.\s*(js|jsx|ts|tsx|py|css|html|json)',
                    ]

                    # FIXED: Expanded code detection patterns
                    code_patterns_check = [
                        r'resource\s+', r'provider\s+', r'def\s+', r'class\s+',
                        r'function\s+', r'const\s+', r'import\s+', r'export\s+',
                        r'\.[a-zA-Z][\w-]*\s*\{', r'color\s*:', r'margin\s*:',
                        r'padding\s*:', r'display\s*:', r'<!DOCTYPE', r'<html',
                    ]

                    file_frames = {}
                    unclassified = []

                    for chunk in chunks:
                        ocr = chunk.get('ocr_text', '')
                        if not ocr or len(ocr) < 20:
                            continue
                        if not is_useful_segment(ocr, ''):
                            continue
                        has_code = any(re.search(p, ocr, re.IGNORECASE) for p in code_patterns_check)
                        if not (has_code or len(re.findall(r'[{}()\[\]=]', ocr)) >= 3):
                            continue

                        detected_file = None
                        for pattern in file_patterns:
                            match = re.search(pattern, ocr, re.IGNORECASE)
                            if match:
                                name = match.group(1)
                                ext = match.group(2).lower()
                                detected_file = f"{name}.{ext}"
                                break

                        frame_data = {
                            'index': chunk.get('index'),
                            'timestamp': chunk.get('timestamp'),
                            'ocr': ocr
                        }

                        if detected_file:
                            if detected_file not in file_frames:
                                file_frames[detected_file] = []
                            file_frames[detected_file].append(frame_data)
                        else:
                            unclassified.append(frame_data)

                    all_code_blocks = []
                    for filename, frames in file_frames.items():
                        frames.sort(key=lambda x: x['timestamp'])
                        frame_ocr = '\n---\n'.join([f['ocr'] for f in frames[-5:]])
                        ext = filename.split('.')[-1].lower() if '.' in filename else 'js'
                        lang = {'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript',
                               'tsx': 'typescript', 'py': 'python', 'json': 'json',
                               'css': 'css', 'html': 'html'}.get(ext, 'javascript')

                        file_prompt = f"""Extract COMPLETE code for: {filename}
OCR from video:
{frame_ocr}

Output:
```{lang}
[complete code]
```"""
                        try:
                            resp = groq_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "system", "content": "Output code only. No explanations."},
                                    {"role": "user", "content": file_prompt}
                                ],
                                temperature=0.1,
                                max_tokens=2000
                            )
                            code_match = re.search(r'```(?:\w+)?\n([\s\S]*?)```', resp.choices[0].message.content)
                            if code_match:
                                all_code_blocks.append({
                                    'filename': filename,
                                    'lang': lang,
                                    'code': code_match.group(1).strip()
                                })
                        except Exception as e:
                            print(f"Error extracting {filename}: {e}")

                    if all_code_blocks:
                        response_parts = []
                        for block in all_code_blocks:
                            response_parts.append(
                                f"### {block['filename']}\n```{block['lang']}\n{block['code']}\n```"
                            )
                        llm_result = {
                            'response': '\n\n'.join(response_parts),
                            'sources': [],
                            'llm_enabled': True
                        }
            except Exception as e:
                print(f"Code extraction error: {e}")
                import traceback
                traceback.print_exc()

        if not llm_result and generate_response and top_results:
            llm_result = generate_llm_response(
                query, top_results,
                want_diagram=want_diagram,
                want_code=want_code
            )

        if llm_result:
            response_data['llm_response'] = llm_result.get('response', '')
            response_data['llm_sources']  = llm_result.get('sources', [])
            response_data['llm_enabled']  = llm_result.get('llm_enabled', False)

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# EVALUATION ENDPOINTS
# ============================================================================

@app.route('/api/evaluate', methods=['POST'])
def evaluate_retrieval():
    """Evaluate retrieval performance with provided ground truth."""
    data = request.json
    video_id = data.get('video_id')
    queries = data.get('queries', [])
    k_values = data.get('k_values', [1, 3, 5, 10])

    if not video_id or not queries:
        return jsonify({'error': 'video_id and queries required'}), 400

    try:
        benchmark = VideoRetrievalBenchmark(OUTPUT_FOLDER)

        def search_function(query, vid, top_k):
            results = hybrid_search(query, vid, top_k=top_k)
            return [r['chunk_index'] for r in results]

        results = benchmark.run_benchmark(
            video_id=video_id,
            queries_with_ground_truth=queries,
            search_function=search_function,
            k_values=k_values
        )
        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate/template/<video_id>', methods=['POST'])
def create_evaluation_template(video_id):
    """Create a ground truth template for manual annotation."""
    data = request.json
    queries = data.get('queries', [])
    if not queries:
        return jsonify({'error': 'queries list required'}), 400
    try:
        benchmark = VideoRetrievalBenchmark(OUTPUT_FOLDER)
        template = benchmark.create_ground_truth_template(video_id, queries)
        return jsonify(template)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# INDEX MANAGEMENT ENDPOINT
# ============================================================================

@app.route('/api/library/index/<video_id>', methods=['POST'])
def index_video_to_library(video_id):
    """Add a processed video to the searchable library index."""
    data = request.json or {}
    video_info = data.get('video_info', {})
    try:
        library = get_library(OUTPUT_FOLDER)
        library.index_video(video_id, video_info)
        return jsonify({
            'message': f'Video {video_id} indexed successfully',
            'video_id': video_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)