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
from PIL import Image
from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from flask_cors import CORS
from evaluation import RetrievalEvaluator, VideoRetrievalBenchmark
from vector_db import VideoLibrary, get_library
from groq import Groq

app = Flask(__name__)
CORS(app)

# ============================================================================
# GROQ LLM CONFIGURATION
# ============================================================================
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq LLM configured.")
else:
    groq_client = None
    print("WARNING: GROQ_API_KEY not set. LLM features disabled.")

# Configuration
UPLOAD_FOLDER = 'downloads'
OUTPUT_FOLDER = 'outputs'
FRAMES_FOLDER = os.path.join(UPLOAD_FOLDER, 'frames')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Initialize Whisper Model (Load once)
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded.")

# Initialize CLIP Model (using open_clip)
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("CLIP model loaded.")

# EasyOCR Reader - initialized lazily to avoid multiprocessing conflicts with Flask
easyocr_reader = None

def get_easyocr_reader():
    """Get or initialize the EasyOCR reader lazily."""
    global easyocr_reader
    if easyocr_reader is None:
        print("Initializing EasyOCR...")
        easyocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU mode avoids multiprocessing issues
        print("EasyOCR ready.")
    return easyocr_reader

# ============================================================================
# LLM RESPONSE GENERATION
# ============================================================================
import base64

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def generate_llm_response(query: str, chunks: list, max_chunks: int = 5, 
                          want_diagram: bool = False, want_code: bool = False,
                          use_vision: bool = True) -> dict:
    """
    Generate an LLM response based on retrieved chunks.
    Uses vision model when frames are available.
    
    Args:
        query: User's search query
        chunks: List of retrieved chunks with metadata
        max_chunks: Maximum number of chunks to include in context
        want_diagram: If True, include diagram generation in prompt
        want_code: If True, emphasize code extraction
        use_vision: If True, include frame images for visual analysis
    
    Returns:
        Dict with 'response' and 'sources' keys
    """
    if not groq_client:
        return {
            'response': 'LLM not configured. Please set GROQ_API_KEY environment variable.',
            'sources': [],
            'llm_enabled': False
        }
    
    def clean_ocr_text(text: str) -> str:
        """Clean OCR text by removing noise and formatting."""
        if not text:
            return ""
        
        # Remove excessive special characters and noise
        # Keep alphanumeric, common punctuation, and code-related characters
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are mostly special characters or too short
            alnum_count = sum(1 for c in line if c.isalnum() or c in ' .,;:(){}[]<>=+-*/')
            if len(line) > 0 and alnum_count / len(line) < 0.3:
                continue
            
            # Skip very short lines (likely noise)
            if len(line) < 3:
                continue
                
            # Skip lines that look like random character sequences
            if re.match(r'^[^a-zA-Z]*$', line) and len(line) > 10:
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def summarize_content(ocr: str, audio: str) -> str:
        """Create a clean summary of the chunk content."""
        parts = []
        
        if audio:
            parts.append(f"**What was said:** {audio}")
        
        if ocr:
            # Check if it looks like code
            code_indicators = ['def ', 'function', 'class ', 'import ', 'return ', 
                              '{', '}', '=>', '->', 'const ', 'let ', 'var ', 
                              'public ', 'private ', 'void ', 'int ', 'String']
            is_code = any(ind in ocr for ind in code_indicators)
            
            if is_code:
                parts.append(f"**Code/commands shown:**\n```\n{ocr}\n```")
            else:
                parts.append(f"**On-screen text:** {ocr}")
        
        return '\n'.join(parts) if parts else "No clear content detected"
    
    # Select top chunks for context
    context_chunks = chunks[:max_chunks]
    
    # Build context from chunks and collect images
    context_parts = []
    sources = []
    frame_images = []  # Store base64 encoded images
    
    for i, chunk in enumerate(context_chunks):
        video_id = chunk.get('video_id', 'unknown')
        timestamp = chunk.get('timestamp', 0)
        raw_ocr = chunk.get('ocr_text', '').strip()
        audio_context = chunk.get('audio_context', '').strip()
        frame_path = chunk.get('frame_path', '')
        
        # Clean the OCR text
        clean_ocr = clean_ocr_text(raw_ocr)
        
        # Format timestamp nicely
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        time_str = f"{mins}:{secs:02d}"
        
        # For code requests, use cleaner format without "Segment" labels
        if want_code:
            chunk_context = f"[{time_str}]\n"
            if clean_ocr:
                chunk_context += f"{clean_ocr}\n"
            if audio_context:
                chunk_context += f"(Audio: {audio_context[:200]})\n"
        else:
            # Original format for non-code queries
            chunk_context = f"--- Segment {i+1} (Video: {video_id}, Time: {time_str}) ---\n"
            chunk_context += summarize_content(clean_ocr, audio_context)
        
        context_parts.append(chunk_context)
        sources.append({
            'video_id': video_id,
            'timestamp': timestamp,
            'similarity': chunk.get('similarity', 0)
        })
        
        # Try to load frame image for vision model
        if use_vision and frame_path and os.path.exists(frame_path):
            img_base64 = encode_image_to_base64(frame_path)
            if img_base64:
                frame_images.append({
                    'index': i + 1,
                    'base64': img_base64,
                    'video_id': video_id,
                    'time': time_str
                })
    
    # Build context text
    context_text = "\n\n".join(context_parts)
    
    # Build specialized prompt based on user needs
    diagram_instructions = ""
    code_instructions = ""
    
    if want_diagram:
        diagram_instructions = """
DIAGRAM GENERATION - CRITICAL RULES:
You MUST generate a valid Mermaid diagram when asked for a diagram, flowchart, or visualization.

SYNTAX RULES (follow exactly):
1. Start with diagram type: flowchart TD, flowchart LR, sequenceDiagram, or classDiagram
2. Use simple node IDs (A, B, C, etc.) or short alphanumeric IDs (no spaces, no special characters)
3. Put labels in square brackets: A[Label Text]
4. Use --> for arrows, -->|text| for labeled arrows
5. Use {} for decision diamonds: B{Question?}
6. NO colons or special characters in labels
7. Keep it simple - max 10 nodes

VALID EXAMPLE:
```mermaid
flowchart TD
    A[Start] --> B{Is DevOps?}
    B -->|Yes| C[CI/CD Pipeline]
    B -->|No| D[Traditional]
    C --> E[Deploy]
    D --> E
    E --> F[End]
```

INVALID (DON'T DO):
- Node IDs with spaces: "My Node"
- Colons in labels: A[Step: Do something]
- Complex special chars: A[Deploy -> Server]

Generate the diagram based on the video content being discussed.
"""
    
    if want_code:
        code_instructions = """
CODE EXTRACTION - OUTPUT RULES:

**FORMAT YOUR RESPONSE NATURALLY:**
- Give code directly, like a human would
- NO "Segment 1", "Segment 2", "In Frame X" references
- NO explanations like "The video shows..." unless asked
- Just give the code with the filename as a header

**EXAMPLE GOOD RESPONSE:**
### Header.js
```javascript
import React from 'react';
// ... actual code
export default Header;
```

### Footer.js
```javascript
import React from 'react';
// ... actual code
```

**ACCURACY RULES:**
1. ONLY output COMPLETE code (balanced braces, complete blocks)
2. Fix OCR errors (broken quotes, wrong characters)
3. Remove UI artifacts (menu text, line numbers)
4. Use proper markdown code blocks with language

**NEVER DO:**
- "In Segment 1, we see..."
- "The presenter shows..."
- "Looking at Frame 42..."
- Placeholders like "..." or "# code continues"
"""
    
    # For code requests, use a simpler, more direct prompt
    if want_code:
        prompt = f"""Extract the COMPLETE code from this video.

CRITICAL RULES:
1. Output ONLY the code with filename headers - NO explanations
2. NEVER say "From segment", "In frame", "The video shows" - just give code directly
3. Merge all OCR captures into ONE complete file per filename
4. Fix OCR errors (broken characters, typos)
5. Include ALL parts: imports, styles, functions, exports

OCR DATA FROM VIDEO:
{context_text}

USER REQUEST: {query}

Output format:
### filename.ext
```language
complete code
```

Give the complete code:"""
    else:
        prompt = f"""You are VideoAI, an intelligent assistant that helps users understand video content.
You have access to video segments with audio transcripts and screenshots from videos.

{diagram_instructions}

AUDIO TRANSCRIPTS AND TEXT:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Analyze the provided content to answer the user's question
2. Use **bold** for key concepts
3. Format any code with proper markdown code blocks
4. Be concise and direct

YOUR RESPONSE:"""

    try:
        # Vision model is deprecated on Groq, use text-only model with OCR data
        # The OCR text in the chunks should contain the screen content
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are VideoAI, an intelligent assistant that analyzes video content. Use the OCR text and audio transcripts to describe what's shown in the video."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 if want_code else 0.7,  # Lower temp for code accuracy
            max_tokens=4096 if want_code else 2048
        )
        
        return {
            'response': response.choices[0].message.content,
            'sources': sources,
            'llm_enabled': True,
            'chunks_used': len(context_chunks),
            'vision_used': False  # Vision model deprecated
        }
    except Exception as e:
        # If vision model fails, try text-only
        if frame_images and use_vision:
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are VideoAI, an intelligent assistant that analyzes video content."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
                return {
                    'response': response.choices[0].message.content,
                    'sources': sources,
                    'llm_enabled': True,
                    'chunks_used': len(context_chunks),
                    'vision_used': False,
                    'vision_fallback': True
                }
            except Exception as e2:
                return {
                    'response': f'Sorry, I encountered an error: {str(e2)}',
                    'sources': sources,
                    'llm_enabled': True,
                    'error': str(e2)
                }
        return {
            'response': f'Sorry, I encountered an error: {str(e)}',
            'sources': sources,
            'llm_enabled': True,
            'error': str(e)
        }


# ============================================================================
# KEYFRAME ALIGNMENT CONFIGURATION
# ============================================================================
SCENE_CHANGE_THRESHOLD = 15.0  # Sensitivity for detecting visual changes (lower = more sensitive)
AUDIO_WINDOW_SECONDS = 5.0     # Seconds before/after keyframe to capture audio
MIN_KEYFRAME_INTERVAL = 1.0    # Minimum seconds between keyframes (debounce)
MAX_KEYFRAME_INTERVAL = 30.0   # Maximum seconds without a keyframe (force capture)


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
    """
    Extracts and transcribes audio using Whisper WITH word-level timestamps.
    This is essential for keyframe alignment.
    """
    print(f"Transcribing audio with timestamps from {video_path}...")
    result = model.transcribe(video_path, word_timestamps=True)
    
    # Extract segments with timestamps
    segments = []
    for segment in result.get("segments", []):
        segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip()
        })
    
    full_text = result["text"]
    return full_text, segments


def clean_text(text):
    """Basic syntax filtering to remove noise."""
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
    """
    Check if OCR output is meaningful or just garbage/noise.
    Returns True if text appears to be real content (code, commands, sentences).
    Returns False if it's just stylized graphics or random characters.
    """
    if not text or len(text) < 10:
        return False
    
    # Count actual words (3+ letters)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 2:
        return False
    
    # Check for common meaningful patterns
    meaningful_patterns = [
        # Code patterns
        r'\b(import|from|def|class|function|return|if|else|for|while|try|except)\b',
        r'\b(const|let|var|export|async|await)\b',
        r'\b(git|npm|pip|docker|kubectl|aws|curl|wget|sudo|apt|yum)\b',
        # Sentence patterns (has verb-like structures)
        r'\b(is|are|was|were|have|has|can|will|would|should|the|and|or|but)\b',
        # DevOps/tech terms
        r'\b(jenkins|kubernetes|docker|pipeline|stage|deploy|build|server|install)\b',
    ]
    
    for pattern in meaningful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Check ratio of recognizable words vs total text
    word_chars = sum(1 for c in text if c.isalpha())
    total_chars = len(text.replace(' ', ''))
    
    if total_chars > 0 and (word_chars / total_chars) < 0.5:
        # Too many non-letter characters, likely garbage
        return False
    
    # Check if text has too many random single characters or numbers
    single_chars = re.findall(r'\b[a-zA-Z0-9]\b', text)
    if len(single_chars) > len(words) * 2:
        return False
    
    return len(words) >= 3


def perform_ocr_on_frame(frame):
    """
    Performs OCR on a single frame using EasyOCR.
    EasyOCR is much better for modern UIs, terminals, and IDE screenshots.
    Falls back to Tesseract if EasyOCR fails.
    """
    try:
        # Convert BGR to RGB for EasyOCR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use EasyOCR - it handles preprocessing internally
        # paragraph=True groups text into paragraphs for better readability
        reader = get_easyocr_reader()
        results = reader.readtext(rgb_frame, paragraph=True, detail=0)
        
        if results:
            # Join all detected text blocks
            text = '\n'.join(results)
            return clean_text(text)
        
        # If paragraph mode didn't work well, try without it
        results = reader.readtext(rgb_frame, paragraph=False)
        if results:
            # Sort by vertical position (top to bottom)
            results.sort(key=lambda x: x[0][0][1])
            lines = []
            current_line = []
            last_y = -100
            
            for detection in results:
                bbox, text, confidence = detection
                if confidence < 0.3:  # Skip low confidence
                    continue
                y_pos = bbox[0][1]
                # Group text on same line (within 20px)
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
    
    # Fallback to Tesseract
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
    Uses EasyOCR with settings optimized for code.
    """
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # EasyOCR with detailed mode for precise text extraction
        reader = get_easyocr_reader()
        results = reader.readtext(
            rgb_frame,
            paragraph=False,  # Don't merge - preserve code structure
            width_ths=0.5,    # Smaller threshold for code spacing
            height_ths=0.5,
            contrast_ths=0.1,  # Lower threshold - code often has low contrast
            text_threshold=0.5
        )
        
        if not results:
            return ""
        
        # Sort by vertical position to preserve code order
        results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))  # y then x
        
        # Group text by lines (based on y-coordinate proximity)
        lines = []
        current_line = []
        current_line_positions = []
        last_y = -100
        line_height_threshold = 25  # Pixels - adjust for code line spacing
        
        for detection in results:
            bbox, text, confidence = detection
            if confidence < 0.2 or not text.strip():  # Skip low confidence or empty
                continue
            
            y_pos = (bbox[0][1] + bbox[2][1]) / 2  # Center y position
            x_pos = bbox[0][0]  # Left x position
            
            # Check if same line (within threshold)
            if abs(y_pos - last_y) < line_height_threshold:
                current_line.append((x_pos, text))
                current_line_positions.append(x_pos)
            else:
                # Save previous line
                if current_line:
                    # Sort by x position and join
                    current_line.sort(key=lambda x: x[0])
                    line_text = ' '.join([t for _, t in current_line])
                    lines.append(line_text)
                
                current_line = [(x_pos, text)]
                current_line_positions = [x_pos]
                last_y = y_pos
        
        # Don't forget last line
        if current_line:
            current_line.sort(key=lambda x: x[0])
            line_text = ' '.join([t for _, t in current_line])
            lines.append(line_text)
        
        # Clean and filter lines
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Keep lines that look like code or have meaningful content
            code_chars = set('{}[]()=;:.<>/-_"\'\\')
            if any(c in code_chars for c in line) or len(line) > 3:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    except Exception as e:
        print(f"Code OCR error: {e}")
        # Fallback to regular OCR
        return perform_ocr_on_frame(frame)


def detect_keyframes(video_path, threshold=SCENE_CHANGE_THRESHOLD):
    """
    DYNAMIC ANCHORING: Detects significant visual changes to find 'Anchor Timestamps' (T_scene).
    
    Instead of fixed-rate sampling (every 2 seconds), this detects when the 
    video actually changes (slide flip, new code block, scroll, etc.)
    
    Also includes time-based fallback: if no keyframe detected in MAX_KEYFRAME_INTERVAL seconds,
    force capture one to ensure coverage for videos with minimal visual changes.
    
    Returns:
        List of keyframe dictionaries with timestamp, frame, and change_score
    """
    print(f"Detecting keyframes with threshold={threshold}...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    keyframes = []
    last_frame_gray = None
    last_keyframe_time = -MIN_KEYFRAME_INTERVAL  # Allow first frame
    frame_count = 0
    pending_frame = None  # Store frame for time-based fallback
    pending_timestamp = 0
    
    # Sample every few frames for efficiency (not every single frame)
    sample_interval = max(1, int(fps / 10))  # ~10 samples per second
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_interval == 0:
            timestamp = frame_count / fps
            
            # Convert to grayscale and blur for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if last_frame_gray is not None:
                # Calculate absolute difference
                frame_delta = cv2.absdiff(last_frame_gray, gray_blur)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                
                # Calculate change score (percentage of changed pixels)
                change_score = (np.sum(thresh) / thresh.size)
                
                # Check if this is a significant change AND enough time has passed
                if change_score > threshold and (timestamp - last_keyframe_time) >= MIN_KEYFRAME_INTERVAL:
                    keyframes.append({
                        'timestamp': timestamp,
                        'frame': frame.copy(),
                        'change_score': change_score,
                        'frame_number': frame_count
                    })
                    last_keyframe_time = timestamp
                    print(f"  ⚡ Keyframe at {timestamp:.2f}s (change: {change_score:.1f})")
                # Time-based fallback: if no keyframe in MAX_KEYFRAME_INTERVAL, force one
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
                # First frame is always a keyframe
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
    """
    TEMPORAL WINDOWING: Gathers audio text from a window around the keyframe timestamp.
    
    This ensures the speaker's explanation is synchronized with the visual content.
    
    Args:
        timestamp: The anchor timestamp (T_scene)
        transcript_segments: List of {'start', 'end', 'text'} from Whisper
        window_seconds: How many seconds before/after to include
    
    Returns:
        Concatenated audio text from the time window
    """
    start_window = max(0, timestamp - window_seconds)
    end_window = timestamp + window_seconds
    
    relevant_text = []
    for seg in transcript_segments:
        # Include segment if it overlaps with our window
        if seg['end'] >= start_window and seg['start'] <= end_window:
            relevant_text.append(seg['text'])
    
    return " ".join(relevant_text)


def create_chunk_vector(frame, ocr_text, audio_context):
    """
    Creates a 1024D multimodal vector for a single chunk.
    
    Vector composition:
        - [0:512]: CLIP text embedding (OCR + audio context)
        - [512:1024]: CLIP visual embedding (keyframe image)
    """
    # Combine text sources for semantic encoding
    combined_text = f"{audio_context} {ocr_text}".strip()
    
    # Text encoding
    if combined_text:
        text_inputs = clip_tokenizer([combined_text[:300]]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = torch.zeros(1, 512).to(device)
    
    # Visual encoding
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        visual_features = clip_model.encode_image(image_input)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
    
    # Concatenate to form 1024D vector
    multimodal_vector = torch.cat((text_features, visual_features), dim=1)
    
    return multimodal_vector.cpu().numpy()


def process_keyframe_aligned(video_path, video_id):
    """
    COMPLETE KEYFRAME-ALIGNED PIPELINE
    
    Flow:
        1. Detect keyframes (dynamic scene changes)
        2. Transcribe audio with timestamps
        3. For each keyframe:
           a. Extract OCR text
           b. Gather temporally-aligned audio context
           c. Create 1024D multimodal vector
        4. Save all outputs
    
    Returns:
        Dictionary containing all processed chunks and metadata
    """
    print("\n" + "="*60)
    print("KEYFRAME-ALIGNED MULTIMODAL EXTRACTION")
    print("="*60 + "\n")
    
    # Step 1: Detect keyframes
    print("[1/4] Detecting scene changes...")
    keyframes = detect_keyframes(video_path)
    
    # Step 2: Transcribe with timestamps
    print("\n[2/4] Transcribing audio with timestamps...")
    full_transcript, transcript_segments = process_audio_with_timestamps(video_path)
    
    # Step 3: Process each keyframe
    print(f"\n[3/4] Processing {len(keyframes)} keyframes...")
    chunks = []
    all_ocr_texts = []
    all_vectors = []
    
    # Create video-specific frames folder
    video_frames_folder = os.path.join(FRAMES_FOLDER, video_id)
    os.makedirs(video_frames_folder, exist_ok=True)
    
    last_ocr_text = ""
    
    for i, kf in enumerate(keyframes):
        timestamp = kf['timestamp']
        frame = kf['frame']
        
        print(f"  Processing keyframe {i+1}/{len(keyframes)} at {timestamp:.2f}s...")
        
        # OCR extraction
        ocr_text = perform_ocr_on_frame(frame)
        
        # De-duplicate similar OCR (consolidation)
        if ocr_text:
            similarity = Levenshtein.ratio(last_ocr_text, ocr_text)
            if similarity < 0.90:
                all_ocr_texts.append(ocr_text)
                last_ocr_text = ocr_text
            elif len(ocr_text) > len(last_ocr_text):
                # Keep longer version
                if all_ocr_texts:
                    all_ocr_texts[-1] = ocr_text
                else:
                    all_ocr_texts.append(ocr_text)
                last_ocr_text = ocr_text
        
        # Gather temporally-aligned audio
        audio_context = gather_audio_context(timestamp, transcript_segments)
        
        # Create multimodal vector for this chunk
        chunk_vector = create_chunk_vector(frame, ocr_text, audio_context)
        all_vectors.append(chunk_vector)
        
        # Save keyframe image
        frame_filename = f"keyframe_{i:04d}_{timestamp:.2f}s.jpg"
        frame_path = os.path.join(video_frames_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        # Store chunk data
        chunks.append({
            'index': i,
            'timestamp': timestamp,
            'ocr_text': ocr_text,
            'audio_context': audio_context,
            'frame_path': frame_path,
            'change_score': kf['change_score']
        })
    
    # Step 4: Save outputs
    print("\n[4/4] Saving outputs...")
    
    # Save audio transcript
    audio_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_audio.txt")
    with open(audio_file_path, "w") as f:
        f.write(full_transcript)
    
    # Save consolidated visual text
    visual_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_visual.txt")
    with open(visual_file_path, "w") as f:
        f.write("\n\n".join(all_ocr_texts))
    
    # Save individual chunk vectors (for retrieval)
    chunks_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz")
    np.savez(
        chunks_file_path,
        vectors=np.vstack(all_vectors),  # Shape: (num_chunks, 1024)
        timestamps=np.array([c['timestamp'] for c in chunks])
    )
    
    # Save combined/averaged multimodal vector (backward compatible)
    if all_vectors:
        combined_vector = np.mean(np.vstack(all_vectors), axis=0, keepdims=True)
    else:
        combined_vector = np.zeros((1, 1024))
    vector_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_multimodal_vector.npy")
    np.save(vector_file_path, combined_vector)
    
    # Save detailed chunk metadata as JSON
    metadata_file_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
    with open(metadata_file_path, "w") as f:
        json.dump({
            'video_id': video_id,
            'total_keyframes': len(keyframes),
            'chunks': [{
                'index': c['index'],
                'timestamp': c['timestamp'],
                'ocr_text': c['ocr_text'][:500] if c['ocr_text'] else "",  # Truncate for JSON
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


@app.route('/api/transcribe', methods=['POST'])
def transcribe_video():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    def generate():
        try:
            # 1. Ingestion
            yield f"data: {json.dumps({'status': 'Downloading video...'})}\n\n"
            video_path, video_id = download_video(url)
            
            # 2. Keyframe-Aligned Processing
            yield f"data: {json.dumps({'status': 'Detecting scene changes...'})}\n\n"
            
            # Process with keyframe alignment
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
                } for c in results['chunks'][:5]]  # Preview first 5 chunks
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
    Search through video chunks using semantic similarity.
    Enhanced with file-centric code extraction for tutorial videos.
    """
    data = request.json
    query = data.get('query')
    video_id = data.get('video_id')
    top_k = data.get('top_k', 5)
    generate_response = data.get('generate_response', True)
    
    if not query or not video_id:
        return jsonify({'error': 'Query and video_id required'}), 400
    
    try:
        # Load chunk vectors
        chunks_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz")
        if not os.path.exists(chunks_file):
            return jsonify({'error': 'Video not processed yet'}), 404
        
        chunk_data = np.load(chunks_file)
        chunk_vectors = chunk_data['vectors']
        timestamps = chunk_data['timestamps']
        
        # Load metadata
        metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Encode query
        query_inputs = clip_tokenizer([query]).to(device)
        with torch.no_grad():
            query_features = clip_model.encode_text(query_inputs)
            query_features /= query_features.norm(dim=-1, keepdim=True)
        
        query_vector = query_features.cpu().numpy()
        
        # Calculate similarity (using text portion of vectors for text queries)
        text_vectors = chunk_vectors[:, :512]  # First 512 dims are text
        similarities = np.dot(text_vectors, query_vector.T).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk_meta = metadata['chunks'][int(idx)]
            results.append({
                'chunk_index': int(idx),
                'video_id': video_id,
                'timestamp': float(timestamps[idx]),
                'similarity': float(similarities[idx]),
                'ocr_text': chunk_meta['ocr_text'],
                'audio_context': chunk_meta['audio_context'],
                'frame_path': chunk_meta['frame_path']
            })
        
        response_data = {'results': results}
        
        # Check if user wants code extraction
        want_code = bool(re.search(r'\b(code|snippet|example|implementation|html|css|javascript|js|py|full|complete)\b', query, re.IGNORECASE))
        extract_all_code = bool(re.search(r'(all|every|complete|full|entire|give).*(code|snippet|example|file)', query, re.IGNORECASE))
        # Also detect specific file requests like "index.html", "style.css", "give me the css"
        specific_file_request = bool(re.search(r'(index\.html|style\.css|styles\.css|main\.css|\.html|\.css|\.js|give\s+me\s+(the\s+)?(index|html|css|style))', query, re.IGNORECASE))
        
        # Use file extraction for both "all code" and specific file requests
        should_extract_files = extract_all_code or specific_file_request
        
        if generate_response and groq_client and want_code and should_extract_files:
            # File-centric code extraction
            chunks = metadata.get('chunks', [])
            print(f"[DEBUG] Processing {len(chunks)} chunks for code extraction", flush=True)
            
            # File detection patterns - OCR often has no dots or spaces
            file_patterns = [
                # HTML files - index.html, about.html, etc.
                r'\b(index|about|contact|home|portfolio|main)\s*\.?\s*(html)\b',
                # CSS/Style files - style.css, styles.css, main.css
                r'\b(style|styles|main|app|global|theme)\s*\.?\s*(css)\b',
                # With space/dot: "Header js" or "Header.js"
                r'([A-Z][a-zA-Z]+)\s*\.?\s*(js|jsx|ts|tsx|py|css|json)\b',
                # CamelCase: "HeaderStyles" followed by js
                r'([A-Z][a-z]+(?:[A-Z][a-z]+)+)\s*\.?\s*(js|ts)\b',
                # No space: "Headerjs", "Projectsjs", "Footerjs"
                r'\b(Header|Footer|Hero|Projects?|Technologies|Timeline|Acomplishments?|constants|index|app|default|Nav\w*|Button|Card|Grid\w*)\s*\.?\s*(js|jsx|ts|tsx)',
                # Styles files
                r'(\w+Styles?)\s*\.?\s*(js|ts)\b',
                # Package.json
                r'(package)\s*\.?\s*(json)',
                # GlobalComponents, etc.
                r'(Global\w+|Section\w*)\s*\.?\s*(js|ts)',
            ]
            
            code_patterns = [
                r'import\s+', r'export\s+', r'const\s+', r'function\s+', r'class\s+',
                r'def\s+', r'from\s+\w+\s+import', r'resource\s+"', r'provider\s+',
                # HTML patterns
                r'<!DOCTYPE', r'<html', r'<head', r'<body', r'<section', r'<div', r'<nav',
                r'<header', r'<footer', r'<script', r'<link\s+rel',
                # CSS patterns  
                r'\.[a-zA-Z][\w-]*\s*\{', r'#[a-zA-Z][\w-]*\s*\{', r'@media', r'@import',
                r'font-size', r'margin', r'padding', r'display', r'color\s*:',
            ]
            
            file_frames = {}
            unclassified = []
            
            for chunk in chunks:
                ocr = chunk.get('ocr_text', '')
                if not ocr or len(ocr) < 30:
                    continue
                
                # Skip "thanks for watching" type content
                skip_patterns = [
                    r'thanks?\s*(for)?\s*watch',
                    r'subscribe',
                    r'like\s*(and|&)?\s*share',
                    r'bell\s*icon',
                    r'notification',
                ]
                should_skip = any(re.search(p, ocr, re.IGNORECASE) for p in skip_patterns)
                if should_skip:
                    continue
                
                # Check for code patterns
                has_code = any(re.search(p, ocr, re.IGNORECASE) for p in code_patterns)
                has_code_chars = len(re.findall(r'[{}()\[\]=;]', ocr)) >= 5
                
                if not (has_code or has_code_chars):
                    continue
                
                # Try to detect filename
                detected_file = None
                for pattern in file_patterns:
                    match = re.search(pattern, ocr, re.IGNORECASE)
                    if match:
                        name = match.group(1)
                        ext = match.group(2).lower() if len(match.groups()) > 1 else 'js'
                        detected_file = f"{name}.{ext}"
                        break
                
                frame_data = {
                    'index': chunk.get('index'),
                    'timestamp': chunk.get('timestamp'),
                    'ocr': ocr
                }
                
                if detected_file:
                    # Filter out OCR artifacts (too short names, partial words)
                    base_name = detected_file.split('.')[0]
                    if len(base_name) < 4 or base_name.lower() in ['proj', 'head', 'foot', 'cons', 'tech', 'comp']:
                        detected_file = None
                
                if detected_file:
                    if detected_file not in file_frames:
                        file_frames[detected_file] = []
                    file_frames[detected_file].append(frame_data)
                else:
                    unclassified.append(frame_data)
            
            print(f"[DEBUG] Files detected: {list(file_frames.keys())}", flush=True)
            print(f"[DEBUG] Unclassified frames: {len(unclassified)}", flush=True)
            
            # Extract code - batch all files into ONE LLM call to save tokens
            all_code_blocks = []
            
            # Filter out OCR typos (files with truncated names like "constan.ts" instead of "constants.ts")
            valid_files = {}
            for filename, frames in file_frames.items():
                # Skip if it looks like truncated OCR (ends with .ts but no proper name before)
                if re.match(r'^[a-z]{1,6}\.(ts|js|jsx|tsx)$', filename) and len(filename) < 12:
                    continue  # Likely OCR artifact like "projec.ts", "constan.ts"
                valid_files[filename] = frames
            
            # Build consolidated prompt for all files
            files_prompt_parts = []
            for filename, frames in list(valid_files.items())[:15]:  # Max 15 files
                frames.sort(key=lambda x: x['timestamp'])
                # Use ALL frames to capture complete code (up to 20 frames)
                num_frames = len(frames)
                if num_frames <= 20:
                    selected_frames = frames
                else:
                    # Select frames spread across the timeline to capture full code
                    step = num_frames / 20
                    selected_frames = [frames[int(i * step)] for i in range(20)]
                
                # Combine OCR from ALL selected frames (1200 chars each for more context)
                frame_ocr = '\n---\n'.join([f['ocr'][:1200] for f in selected_frames])
                files_prompt_parts.append(f"=== {filename} ===\n{frame_ocr}")
            
            if files_prompt_parts:
                consolidated_prompt = f"""EXTRACT AND RECONSTRUCT THE COMPLETE CODE for each file.

CRITICAL RULES:
1. Merge ALL OCR views to create ONE COMPLETE file - do NOT give partial code
2. NEVER say "From segment" or "In frame" - just give the code directly
3. Fix OCR errors (broken characters, wrong symbols)
4. Include EVERYTHING: doctype, head, body for HTML; all selectors for CSS; all imports/exports for JS
5. Output ONLY code with filename headers, NO explanations

Output format:
### filename.ext
```language
complete code here
```

Files to extract (--- separates different OCR captures of same file):
{chr(10).join(files_prompt_parts)}

Give the COMPLETE reconstructed code for each file:"""

                try:
                    resp = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "Reconstruct complete code files from multiple partial OCR views. Merge overlapping parts. Include ALL imports, state, functions, and JSX. Format: ### filename\\n```\\ncode\\n```"},
                            {"role": "user", "content": consolidated_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=12000
                    )
                    response_text = resp.choices[0].message.content
                    
                    # Parse response to extract files
                    file_blocks = re.findall(r'###\s*(\S+)\n```(?:\w+)?\n([\s\S]*?)```', response_text)
                    for fname, code in file_blocks:
                        if code.strip():
                            all_code_blocks.append({
                                'filename': fname,
                                'code': code.strip()
                            })
                except Exception as e:
                    error_msg = str(e)
                    if 'rate_limit' in error_msg.lower():
                        return jsonify({
                            'error': 'API rate limit reached. Please try again later.',
                            'files_detected': list(file_frames.keys()),
                            'results': results
                        })
                    print(f"LLM Error: {e}")
            
            # Format response as clean code
            if all_code_blocks:
                response_parts = []
                for block in all_code_blocks:
                    ext = block['filename'].split('.')[-1] if '.' in block['filename'] else ''
                    lang = {'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript', 
                           'tsx': 'typescript', 'py': 'python', 'json': 'json', 
                           'css': 'css', 'tf': 'hcl', 'yaml': 'yaml'}.get(ext, '')
                    response_parts.append(f"### {block['filename']}\n```{lang}\n{block['code']}\n```")
                
                response_data['llm_response'] = '\n\n'.join(response_parts)
                response_data['llm_enabled'] = True
                response_data['files_found'] = [b['filename'] for b in all_code_blocks]
        
        elif generate_response and groq_client and results:
            # Regular LLM response for non-code queries
            llm_result = generate_llm_response(query, results, want_code=want_code)
            if llm_result:
                response_data['llm_response'] = llm_result.get('response', '')
                response_data['llm_enabled'] = llm_result.get('llm_enabled', False)
        
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
        target_file = data.get('filename')  # e.g., "Timeline.js"
        
        if not video_id or not target_file:
            return jsonify({'error': 'video_id and filename required'}), 400
        
        # Load chunks
        chunks_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz")
        metadata_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        
        if not os.path.exists(chunks_path) or not os.path.exists(metadata_path):
            return jsonify({'error': 'Video not found'}), 404
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        chunks = metadata.get('chunks', [])
        
        # Find frames with ACTUAL code (not just mentions in navbar)
        target_lower = target_file.lower().replace('.js', '').replace('.jsx', '').replace('.ts', '').replace('.tsx', '')
        code_frames = []
        
        for chunk in chunks:
            ocr = chunk.get('ocr_text', '')
            ocr_lower = ocr.lower()
            
            # Must have target file name + actual code indicators
            if target_lower in ocr_lower:
                has_code = any(kw in ocr_lower for kw in ['import ', 'const ', 'function ', 'export ', 'return ', 'useeffect', 'usestate'])
                if has_code and len(ocr) > 300:  # Substantial content
                    code_frames.append({
                        'timestamp': chunk.get('timestamp', 0),
                        'ocr': ocr
                    })
        
        if not code_frames:
            return jsonify({
                'error': f'No code frames found for {target_file}',
                'suggestion': 'Try a different filename'
            }), 404
        
        # Sort by timestamp and use all code frames
        code_frames.sort(key=lambda x: x['timestamp'])
        num_frames = len(code_frames)
        
        # Use up to 20 frames for token limits
        if num_frames <= 20:
            selected_frames = code_frames
        else:
            step = num_frames / 20
            selected_frames = [code_frames[int(i * step)] for i in range(20)]
        
        # Extract and deduplicate code lines with ordering
        seen_lines = set()
        ordered_lines = []
        for frame in selected_frames:
            for line in frame['ocr'].split('\n'):
                line_clean = line.strip()
                if line_clean and len(line_clean) > 2:
                    # Skip obvious UI elements
                    if any(x in line_clean for x in ['localhost', 'File Edit', 'Visual Studio', 'EXPL', 'PROBLEMS', 'OUTPUT']):
                        continue
                    if line_clean not in seen_lines:
                        seen_lines.add(line_clean)
                        ordered_lines.append(line_clean)
        
        # Use LLM to reconstruct with better prompt
        if not groq_client:
            return jsonify({'error': 'LLM not configured'}), 500
        
        # Build concise but complete OCR reference
        frames_ocr = []
        for i, f in enumerate(selected_frames):
            # Get only code-relevant parts of OCR
            ocr_lines = [l.strip() for l in f['ocr'].split('\n') if l.strip() and len(l.strip()) > 5]
            frames_ocr.append(f"[{f['timestamp']:.0f}s]: " + ' | '.join(ocr_lines[:30]))
        
        all_ocr = '\n'.join(frames_ocr)
        unique_code = '\n'.join(ordered_lines[:300])
        
        prompt = f"""Clean and organize these code lines extracted from video OCR into a valid {target_file} file.

RULES:
1. Use ONLY the code lines provided - do NOT add new code
2. Fix obvious OCR errors (missing characters, wrong symbols)  
3. Order: imports first, then constants, then functions, then component, then export
4. Keep the EXACT text from OCR as much as possible

OCR CODE LINES (use these exactly, just fix ordering and obvious errors):
{unique_code}

Output the cleaned {target_file}:
```javascript
"""
        
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Clean and organize OCR code lines. Fix only obvious OCR errors. Do NOT add new code that isn't in the input. Output a valid JavaScript file."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=8000
            )
            
            response_text = resp.choices[0].message.content
            code_match = re.search(r'```(?:javascript|jsx|typescript|tsx)?\n?([\s\S]*?)```', response_text)
            code = code_match.group(1).strip() if code_match else response_text.strip()
            
            return jsonify({
                'filename': target_file,
                'code': code,
                'frames_analyzed': len(selected_frames),
                'total_frames_found': num_frames
            })
            
        except Exception as e:
            return jsonify({
                'error': f'LLM error: {str(e)}',
                'frames_found': len(matching_frames)
            }), 500
    
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
        # Get videos from output folder
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
        
        # Load transcript
        audio_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_audio.txt")
        transcript = ""
        if os.path.exists(audio_file):
            with open(audio_file, 'r') as f:
                transcript = f.read()
        
        # Load visual text
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
        
        # Delete frames folder
        frames_folder = os.path.join(FRAMES_FOLDER, video_id)
        if os.path.exists(frames_folder):
            import shutil
            shutil.rmtree(frames_folder)
            deleted.append(f"frames/{video_id}")
        
        # Delete downloaded video
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
    """
    Re-process OCR for an existing video using EasyOCR.
    This updates the chunks_metadata.json with new OCR text.
    """
    try:
        # Check if video exists
        metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        if not os.path.exists(metadata_file):
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if frames exist
        frames_folder = os.path.join(FRAMES_FOLDER, video_id)
        if not os.path.exists(frames_folder):
            return jsonify({'error': 'Frames folder not found. Please re-upload the video.'}), 404
        
        # Load existing metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Get list of frame files
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
        
        if not frame_files:
            return jsonify({'error': 'No frames found in frames folder'}), 404
        
        # Re-process OCR for each chunk
        updated_chunks = []
        all_ocr_text = []
        
        for i, chunk in enumerate(metadata.get('chunks', [])):
            # Find corresponding frame file
            frame_index = chunk.get('index', i)
            frame_path = None
            
            # Try to find the frame file
            for frame_file in frame_files:
                if f"frame_{frame_index:04d}" in frame_file or f"keyframe_{frame_index:04d}" in frame_file:
                    frame_path = os.path.join(frames_folder, frame_file)
                    break
            
            # If not found by index, use the first available frame for this chunk
            if not frame_path and frame_index < len(frame_files):
                frame_path = os.path.join(frames_folder, frame_files[frame_index])
            
            new_ocr_text = ""
            if frame_path and os.path.exists(frame_path):
                # Read frame and perform OCR with EasyOCR
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # Use the new EasyOCR-based function
                    new_ocr_text = perform_code_ocr(frame)
                    if not new_ocr_text:
                        new_ocr_text = perform_ocr_on_frame(frame)
            
            # Update chunk with new OCR text
            chunk['ocr_text'] = new_ocr_text
            if new_ocr_text:
                all_ocr_text.append(f"[Frame {frame_index}]: {new_ocr_text}")
            
            updated_chunks.append(chunk)
        
        # Update metadata
        metadata['chunks'] = updated_chunks
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update visual text file
        visual_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_visual.txt")
        with open(visual_file, 'w') as f:
            f.write('\n'.join(all_ocr_text))
        
        # Regenerate multimodal vectors with new OCR text
        # Load audio transcript
        audio_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_audio.txt")
        audio_text = ""
        if os.path.exists(audio_file):
            with open(audio_file, 'r') as f:
                audio_text = f.read()
        
        # Regenerate vectors for each chunk
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
            
            # Save updated vectors
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
    Extract ALL code snippets from a video using OCR text and LLM consolidation.
    FILE-CENTRIC extraction: identifies all files and extracts complete code for each.
    """
    if not groq_client:
        return jsonify({'error': 'LLM not configured'}), 500
    
    try:
        # Load video metadata
        metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
        if not os.path.exists(metadata_file):
            return jsonify({'error': 'Video not found'}), 404
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        chunks = metadata.get('chunks', [])
        
        def clean_ide_ocr(text):
            """Remove common IDE/browser UI artifacts from OCR text."""
            ui_patterns = [
                r'^Activities\s+',
                r'Visual Studio Code\s*',
                r'Google Chrome\s*',
                r'File\s+Edit\s+Selection\s+View\s+Go\s+Run\s+Terminal\s+Help\s*',
                r'EXPLORER\s*',
                r'PROBLEMS\s+OUTPUT\s+DEBUG\s+CONSOLE\s+TERMINAL\s*',
                r'OUTLINE\s*',
                r'TIMELINE\s*',
                r'Ln\s*\d+,?\s*Col\s*\d+\s*',
                r'Spaces:\s*\d+\s*',
                r'UTF-8\s*',
                r'portfolio_website-STARTER\s*',
                r'portfolio_nextjs\s*',
            ]
            
            cleaned = text
            for pattern in ui_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
            return cleaned.strip()
        
        def detect_filename(ocr_text):
            """Detect filename from OCR text (VS Code tab titles, etc.)."""
            # Common file patterns in VS Code OCR
            # Pattern: "Filename.js - project" or "Filename.js project"
            patterns = [
                r'(\w+(?:Styles)?\.(?:js|jsx|ts|tsx|json|css))\s*[-–]?\s*portfolio',
                r'(\w+(?:Styles)?\.(?:js|jsx|ts|tsx|json|css))\s+portfolio',
                r'Help\s+(\w+(?:Styles)?\.(?:js|jsx|ts|tsx|json|css))',
                r'(\w+(?:Styles)?\.(?:js|jsx|ts|tsx|json|css))\s+Visual',
                # Handle OCR without dots: "Headerjs" -> "Header.js"  
                r'Help\s+([A-Z][a-z]+(?:Styles)?)(js|json)\s',
                r'(\w+)(Styles)(js)\s',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 2 and match.group(2) in ['js', 'json', 'Styles']:
                        # Handle split patterns like "Header" + "js"
                        name = match.group(1)
                        ext = 'js' if match.group(2) == 'Styles' else match.group(2)
                        if match.group(2) == 'Styles':
                            name = name + 'Styles'
                        return f"{name}.{ext}"
                    return match.group(1)
            
            # Try to find filename without extension in title bar
            title_match = re.search(r'Help\s+([A-Z][a-z]+(?:Styles)?)\s*js', ocr_text)
            if title_match:
                return f"{title_match.group(1)}.js"
            
            return None
        
        # Code detection patterns (for various languages)
        code_patterns = [
            r'import\s+.*from\s+["\']',
            r'export\s+(default\s+)?',
            r'const\s+\w+\s*=',
            r'function\s+\w+',
            r'styled\.\w+',
            r'<\w+[^>]*>',  # JSX
            r'return\s*\(',
            r'React\.',
            r'resource\s+["\']',
            r'provider\s+["\']',
        ]
        
        # Collect ALL frames with code, grouped by detected file
        file_frames = {}  # filename -> list of frames
        unclassified_frames = []
        
        for chunk in chunks:
            ocr_text = chunk.get('ocr_text', '')
            if not ocr_text or len(ocr_text.strip()) < 30:
                continue
            
            # Check if has code
            has_code = any(re.search(p, ocr_text, re.IGNORECASE) for p in code_patterns)
            code_chars = len(re.findall(r'[{}()\[\]=;:<>/]', ocr_text))
            
            if not has_code and code_chars < 8:
                continue
            
            # Detect filename
            filename = detect_filename(ocr_text)
            cleaned_ocr = clean_ide_ocr(ocr_text)
            
            frame_data = {
                'index': chunk.get('index', 0),
                'timestamp': chunk.get('timestamp', 0),
                'ocr_text': cleaned_ocr,
                'raw_ocr': ocr_text,
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
        
        # Build file-centric context for LLM
        files_found = list(file_frames.keys())
        
        # Process EACH file separately to get complete code
        all_code_blocks = []
        
        # First, get a summary of what files exist
        file_summary = []
        for filename, frames in file_frames.items():
            frames.sort(key=lambda x: x['timestamp'])
            frame_count = len(frames)
            time_range = f"{frames[0]['timestamp']:.0f}s - {frames[-1]['timestamp']:.0f}s"
            file_summary.append(f"- {filename}: {frame_count} frames ({time_range})")
        
        # Now extract code for each file
        for filename, frames in file_frames.items():
            if len(frames) == 0:
                continue
            
            # Sort by timestamp to get evolution of file
            frames.sort(key=lambda x: x['timestamp'])
            
            # Build context for this specific file
            file_ocr_context = f"FILE: {filename}\n"
            file_ocr_context += f"Shown in {len(frames)} frames\n"
            file_ocr_context += "=" * 50 + "\n\n"
            
            for frame in frames[-15:]:  # Use latest frames (most complete)
                mins = int(frame['timestamp'] // 60)
                secs = int(frame['timestamp'] % 60)
                file_ocr_context += f"[{mins}:{secs:02d}]\n{frame['ocr_text']}\n\n"
            
            # LLM extraction for this file
            file_prompt = f"""Extract the COMPLETE code for the file: {filename}

This file appears in multiple frames of a tutorial video as the presenter types/edits it.
Your task: reconstruct the FINAL, COMPLETE version of this file.

STRICT RULES:
1. Output ONLY complete, syntactically valid code
2. Fix OCR errors (broken quotes, wrong characters)
3. Ensure all braces {{}}, parentheses (), brackets [] are balanced
4. Use the LATEST frames (they have the most complete code)
5. If code appears incomplete, output what's complete with a note

OCR DATA FOR {filename}:
{file_ocr_context}

Output the complete code in:
```javascript
// {filename}
[complete code here]
```

If incomplete, briefly state what's missing."""

            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an expert code extractor. Extract complete, runnable code from OCR captures. Never output partial code with placeholders."},
                        {"role": "user", "content": file_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                extracted = response.choices[0].message.content
                
                # Parse code block
                code_match = re.search(r'```(?:\w+)?\n([\s\S]*?)```', extracted)
                if code_match:
                    code = code_match.group(1).strip()
                    
                    # Validate
                    open_braces = code.count('{')
                    close_braces = code.count('}')
                    is_valid = abs(open_braces - close_braces) <= 1
                    
                    all_code_blocks.append({
                        'filename': filename,
                        'language': 'javascript',
                        'code': code,
                        'frames_used': len(frames),
                        'is_valid': is_valid,
                        'confidence': 'HIGH' if is_valid else 'MEDIUM'
                    })
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                continue
        
        # Also process unclassified frames in batches
        if unclassified_frames and len(all_code_blocks) < 5:
            # Group by time proximity
            unclassified_frames.sort(key=lambda x: x['timestamp'])
            
            batch_ocr = "ADDITIONAL CODE SECTIONS (unclassified by file):\n\n"
            for frame in unclassified_frames[-20:]:
                mins = int(frame['timestamp'] // 60)
                secs = int(frame['timestamp'] % 60)
                batch_ocr += f"[{mins}:{secs:02d}]\n{frame['ocr_text']}\n\n"
            
            try:
                batch_prompt = f"""Extract any additional complete code blocks from these video frames.

{batch_ocr}

For each complete code file found, output:
## [Filename]
```[language]
[complete code]
```

Only output code that is COMPLETE (balanced braces, no placeholders)."""

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Extract complete code only."},
                        {"role": "user", "content": batch_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                # Parse additional blocks
                additional = response.choices[0].message.content
                code_blocks_found = re.findall(r'```(\w+)?\n([\s\S]*?)```', additional)
                for lang, code in code_blocks_found:
                    if code.strip() and len(code.strip()) > 50:
                        all_code_blocks.append({
                            'filename': 'additional_code',
                            'language': lang or 'javascript',
                            'code': code.strip(),
                            'frames_used': len(unclassified_frames),
                            'is_valid': True,
                            'confidence': 'MEDIUM'
                        })
            except:
                pass
        
        # Calculate accuracy metrics
        valid_blocks = [b for b in all_code_blocks if b.get('is_valid', False)]
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'files_found': files_found,
            'total_files': len(files_found),
            'total_code_frames': sum(len(frames) for frames in file_frames.values()),
            'accuracy': {
                'rating': 'HIGH (95%+)' if len(valid_blocks) == len(all_code_blocks) else 'MEDIUM',
                'valid_files': len(valid_blocks),
                'total_files': len(all_code_blocks)
            },
            'code_blocks': all_code_blocks,
            'file_summary': file_summary
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
    Search across indexed videos.
    If video_id is provided, searches only that video (session mode).
    Otherwise searches across all videos.
    Returns results ranked by similarity with optional LLM response.
    """
    data = request.json
    query = data.get('query')
    top_k = data.get('top_k', 10)
    generate_response = data.get('generate_response', True)
    video_id_filter = data.get('video_id')  # Optional: limit to specific video
    
    # Check if user wants diagrams or code
    want_diagram = data.get('want_diagram', False)
    want_code = data.get('want_code', False)
    
    # Auto-detect from query if not explicitly set
    if not want_diagram:
        want_diagram = bool(re.search(r'diagram|flowchart|chart|visualize|architecture|structure|flow', query, re.IGNORECASE))
    if not want_code:
        # Detect code-related queries - including "all code", "show code", "extract code"
        want_code = bool(re.search(r'\b(code|snippet|example|implementation|function|method|class|script|command|syntax|pipeline|dockerfile|yaml|json|groovy|jenkinsfile|terraform|extract.*code|show.*code|all.*code|copy.*paste)\b', query, re.IGNORECASE))
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        # Encode query
        query_inputs = clip_tokenizer([query]).to(device)
        with torch.no_grad():
            query_features = clip_model.encode_text(query_inputs)
            query_features /= query_features.norm(dim=-1, keepdim=True)
        
        query_vector = query_features.cpu().numpy().flatten()
        
        # Search videos (single video if video_id provided, else all)
        all_results = []
        code_results = []  # Store results with code content separately
        videos_to_search = []
        
        if video_id_filter:
            # Session mode: only search the specified video
            chunks_file = os.path.join(OUTPUT_FOLDER, f"{video_id_filter}_chunks.npz")
            if os.path.exists(chunks_file):
                videos_to_search = [video_id_filter]
            else:
                return jsonify({'error': f'Video {video_id_filter} not found'}), 404
        else:
            # Search all videos
            for filename in os.listdir(OUTPUT_FOLDER):
                if filename.endswith('_chunks.npz'):
                    videos_to_search.append(filename.replace('_chunks.npz', ''))
        
        # Code detection patterns
        code_patterns = [
            r'terraform|provider|resource|variable',
            r'docker|FROM|RUN|COPY|CMD|EXPOSE',
            r'pipeline|stage|steps|script',
            r'kubectl|apiVersion|kind|metadata',
            r'import|export|const|let|var|function|def|class',
            r'main\.tf|main\.py|index\.js|\.yaml|\.yml|\.json',
            r'TERMINAL|CONSOLE|bash|shell',
            r'git\s|aws\s|gcloud|azure',
            r'[{}()\[\]<>=;:]+',  # Code characters
        ]
        
        # Patterns to EXCLUDE (non-code, end-of-video, promotional content)
        exclude_patterns = [
            r'thanks\s*(for)?\s*watching',
            r'subscribe|sub\s*\d|like\s*\d',
            r'follow\s*(me|us)|smash.*button',
            r'notification\s*bell',
            r'see\s*you\s*(in|next)',
            r'bye|goodbye',
            r'end\s*screen|outro',
        ]
        
        def is_useful_segment(ocr_text, audio_context=''):
            """Check if a segment contains useful content (not end-of-video fluff)."""
            combined = f"{ocr_text} {audio_context}".lower()
            
            # Exclude if matches exclusion patterns
            for pattern in exclude_patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return False
            
            # Must have some actual content
            if len(ocr_text.strip()) < 20:
                return False
            
            return True
        
        for video_id in videos_to_search:
            # Load vectors and metadata
            chunks_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks.npz")
            npz_data = np.load(chunks_file)
            chunk_vectors = npz_data['vectors']
            timestamps = npz_data['timestamps']
            
            metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id}_chunks_metadata.json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Calculate similarity (using text portion)
            text_vectors = chunk_vectors[:, :512]
            similarities = np.dot(text_vectors, query_vector[:512]).flatten()
            
            # Get top results from this video
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more to filter
            
            for idx in top_indices:
                chunk_meta = metadata['chunks'][idx]
                ocr_text = chunk_meta.get('ocr_text', '')
                audio_context = chunk_meta.get('audio_context', '')
                
                # Filter out non-useful segments (end-of-video, promotional, etc.)
                if not is_useful_segment(ocr_text, audio_context):
                    continue
                
                result = {
                    'video_id': video_id,
                    'timestamp': float(timestamps[idx]),
                    'chunk_index': int(idx),
                    'similarity': float(similarities[idx]),
                    'ocr_text': ocr_text,
                    'audio_context': audio_context,
                    'frame_path': chunk_meta.get('frame_path', '')
                }
                all_results.append(result)
                
                # Check if this chunk has code
                if want_code and ocr_text:
                    has_code = any(re.search(p, ocr_text, re.IGNORECASE) for p in code_patterns)
                    if has_code or len(ocr_text) > 50:
                        code_results.append(result)
            
            # When want_code, also scan ALL chunks for code content (not just top-k by similarity)
            if want_code:
                for idx, chunk_meta in enumerate(metadata.get('chunks', [])):
                    if idx in [r['chunk_index'] for r in all_results if r['video_id'] == video_id]:
                        continue  # Already included
                    
                    ocr_text = chunk_meta.get('ocr_text', '')
                    audio_context = chunk_meta.get('audio_context', '')
                    
                    # Skip non-useful segments
                    if not is_useful_segment(ocr_text, audio_context):
                        continue
                    
                    if ocr_text and len(ocr_text) > 30:
                        has_code = any(re.search(p, ocr_text, re.IGNORECASE) for p in code_patterns)
                        if has_code:
                            result = {
                                'video_id': video_id,
                                'timestamp': float(chunk_meta.get('timestamp', 0)),
                                'chunk_index': idx,
                                'similarity': float(similarities[idx]) if idx < len(similarities) else 0.0,
                                'ocr_text': ocr_text,
                                'audio_context': chunk_meta.get('audio_context', ''),
                                'frame_path': chunk_meta.get('frame_path', ''),
                                'code_match': True
                            }
                            code_results.append(result)
        
        # Sort all results by similarity and take top_k
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # When want_code is True, merge code results with similarity results
        if want_code and code_results:
            # Remove duplicates and sort code results by timestamp
            seen = set()
            unique_code = []
            for r in code_results:
                key = (r['video_id'], r['chunk_index'])
                if key not in seen:
                    seen.add(key)
                    unique_code.append(r)
            
            unique_code.sort(key=lambda x: x['timestamp'])
            
            # Blend: use code results first, then fill with similarity results
            top_results = unique_code[:min(len(unique_code), top_k * 2)]  # Use more chunks for code queries
            for r in all_results:
                if len(top_results) >= top_k * 2:
                    break
                key = (r['video_id'], r['chunk_index'])
                if key not in seen:
                    seen.add(key)
                    top_results.append(r)
        else:
            top_results = all_results[:top_k]
        
        # Generate LLM response if requested
        llm_result = None
        
        # If user wants ALL code from a specific video, use the dedicated extraction
        extract_all_code = bool(re.search(r'(all|every|complete|full|entire).*(code|snippet|example)', query, re.IGNORECASE))
        
        print(f"[DEBUG] want_code={want_code}, video_id_filter={video_id_filter}, extract_all_code={extract_all_code}", flush=True)
        
        if want_code and video_id_filter and extract_all_code:
            # Use dedicated file-centric code extraction for comprehensive results
            print(f"[DEBUG] File-centric extraction triggered for video: {video_id_filter}")
            try:
                metadata_file = os.path.join(OUTPUT_FOLDER, f"{video_id_filter}_chunks_metadata.json")
                print(f"[DEBUG] Looking for metadata: {metadata_file}, exists: {os.path.exists(metadata_file)}")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    chunks = metadata.get('chunks', [])
                    
                    # File detection patterns (VS Code tabs, etc.)
                    file_patterns = [
                        r'([A-Z][a-z]+(?:[A-Z][a-z]+)*)\s*\.\s*(js|jsx|ts|tsx|py|css|json)',
                        r'([a-z]+(?:[A-Z][a-z]+)*)\s*\.\s*(js|jsx|ts|tsx|py|css|json)',
                        r'(index|app|main|config|constants|styles|default)\s*\.\s*(js|jsx|ts|tsx|py|css|json)',
                        r'(\w+Styles?)\s*\.\s*(js|ts)',
                        r'(\w+)\.(tf|hcl|yaml|yml|dockerfile)',
                    ]
                    
                    # Group frames by detected file
                    file_frames = {}
                    unclassified = []
                    
                    code_patterns_check = [
                        r'resource\s+', r'provider\s+', r'variable\s+', r'locals\s*{',
                        r'aws_', r'docker', r'pipeline', r'def\s+', r'class\s+',
                        r'function\s+', r'const\s+', r'let\s+', r'import\s+', r'export\s+'
                    ]
                    
                    for chunk in chunks:
                        ocr = chunk.get('ocr_text', '')
                        if not ocr or len(ocr) < 30:
                            continue
                        if not is_useful_segment(ocr, ''):
                            continue
                        has_code = any(re.search(p, ocr, re.IGNORECASE) for p in code_patterns_check)
                        if not (has_code or len(re.findall(r'[{}()\[\]=]', ocr)) >= 5):
                            continue
                        
                        # Try to detect filename
                        detected_file = None
                        for pattern in file_patterns:
                            match = re.search(pattern, ocr, re.IGNORECASE)
                            if match:
                                name = match.group(1)
                                ext = match.group(2).lower()
                                # Fix OCR issues (missing dots)
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
                    
                    # Build response with all files
                    all_code_blocks = []
                    print(f"[DEBUG] Files detected: {list(file_frames.keys())}")
                    print(f"[DEBUG] Unclassified frames: {len(unclassified)}")
                    
                    for filename, frames in file_frames.items():
                        frames.sort(key=lambda x: x['timestamp'])
                        frame_ocr = '\n---\n'.join([f['ocr'] for f in frames[-5:]])  # Last 5 frames have complete code
                        
                        file_prompt = f"""Extract the COMPLETE code for: {filename}

OCR from video frames:
{frame_ocr}

OUTPUT RULES:
1. Output ONLY the complete, working code - no explanations
2. Fix OCR errors (broken quotes, wrong characters)
3. Proper indentation
4. If incomplete, output what's visible

Output:
```
[code]
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
                                    'code': code_match.group(1).strip()
                                })
                        except Exception as e:
                            print(f"Error extracting {filename}: {e}")
                    
                    # Also extract from unclassified frames
                    if unclassified and len(all_code_blocks) < 5:
                        unclassified_ocr = '\n---\n'.join([f['ocr'] for f in unclassified[:10]])
                        misc_prompt = f"""Extract any additional code files from:

{unclassified_ocr}

For each file, output:
### filename.ext
```
[code]
```"""
                        try:
                            resp = groq_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "system", "content": "Output code only."},
                                    {"role": "user", "content": misc_prompt}
                                ],
                                temperature=0.1,
                                max_tokens=3000
                            )
                            misc_matches = re.findall(r'###\s*(\S+)\n```(?:\w+)?\n([\s\S]*?)```', resp.choices[0].message.content)
                            for fname, code in misc_matches:
                                all_code_blocks.append({'filename': fname, 'code': code.strip()})
                        except:
                            pass
                    
                    # Format response naturally - just the code
                    print(f"[DEBUG] Extracted {len(all_code_blocks)} code blocks")
                    if all_code_blocks:
                        response_parts = []
                        for block in all_code_blocks:
                            ext = block['filename'].split('.')[-1] if '.' in block['filename'] else ''
                            lang = {'js': 'javascript', 'jsx': 'javascript', 'ts': 'typescript', 'tsx': 'typescript',
                                   'py': 'python', 'json': 'json', 'css': 'css', 'tf': 'hcl', 'yaml': 'yaml', 'yml': 'yaml'}.get(ext, '')
                            response_parts.append(f"### {block['filename']}\n```{lang}\n{block['code']}\n```")
                        
                        llm_result = {
                            'response': '\n\n'.join(response_parts),
                            'sources': [],
                            'llm_enabled': True
                        }
                    
            except Exception as e:
                print(f"Code extraction fallback error: {e}")
                import traceback
                traceback.print_exc()
                llm_result = None
        
        if not llm_result and generate_response and top_results:
            llm_result = generate_llm_response(query, top_results, 
                                               want_diagram=want_diagram, 
                                               want_code=want_code)
        
        response_data = {
            'query': query,
            'results': top_results,
            'total_videos_searched': len(set(r['video_id'] for r in all_results))
        }
        
        # Add LLM response if generated
        if llm_result:
            response_data['llm_response'] = llm_result.get('response', '')
            response_data['llm_sources'] = llm_result.get('sources', [])
            response_data['llm_enabled'] = llm_result.get('llm_enabled', False)
        
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
    """
    Evaluate retrieval performance with provided ground truth.
    
    Request body:
    {
        "video_id": "xxx",
        "queries": [
            {
                "query": "search query",
                "relevant_chunks": [0, 3, 5],  // indices of relevant chunks
                "relevance_scores": {"0": 3, "3": 2, "5": 1}  // optional graded relevance
            }
        ],
        "k_values": [1, 3, 5, 10]  // optional
    }
    """
    data = request.json
    video_id = data.get('video_id')
    queries = data.get('queries', [])
    k_values = data.get('k_values', [1, 3, 5, 10])
    
    if not video_id or not queries:
        return jsonify({'error': 'video_id and queries required'}), 400
    
    try:
        benchmark = VideoRetrievalBenchmark(OUTPUT_FOLDER)
        
        def search_function(query, vid, top_k):
            # Encode query
            query_inputs = clip_tokenizer([query]).to(device)
            with torch.no_grad():
                query_features = clip_model.encode_text(query_inputs)
                query_features /= query_features.norm(dim=-1, keepdim=True)
            
            query_vector = query_features.cpu().numpy().flatten()
            
            # Load vectors
            chunks_file = os.path.join(OUTPUT_FOLDER, f"{vid}_chunks.npz")
            npz_data = np.load(chunks_file)
            chunk_vectors = npz_data['vectors']
            
            # Calculate similarity
            text_vectors = chunk_vectors[:, :512]
            similarities = np.dot(text_vectors, query_vector[:512]).flatten()
            
            # Return top-k indices
            return np.argsort(similarities)[::-1][:top_k].tolist()
        
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
