import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
import argparse
import logging
import json
import difflib
import os
import pdfplumber
import docx
import csv
import re
from PIL import Image
import pytesseract
import io

# === AUDIO FEATURE IMPORTS ===
import speech_recognition as sr
from gtts import gTTS
import tempfile
import pyaudio
import wave
import threading
import time

# 1. Parse Arguments
parser = argparse.ArgumentParser(description="Run unified chatbot")
parser.add_argument("--model_dir", type=str, default="./outputs/final_model", help="Path to fine-tuned model dir")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Load Tokenizer
logger.info("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
except Exception as e:
    logger.warning(f"Could not load tokenizer from {args.model_dir}, trying a default. Error: {e}")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 3. Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 4. Load Fine-tuned Model (with LoRA)
logger.info("Loading fine-tuned model...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        "teknium/OpenHermes-2.5-Mistral-7B",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    finetuned_model = PeftModel.from_pretrained(
        base_model,
        args.model_dir
    )
    finetuned_model.eval()
    model_loaded = True
except Exception as e:
    logger.error(f"Failed to load fine-tuned model: {e}")
    logger.error("Please ensure --model_dir points to a valid fine-tuned model directory.")
    model_loaded = False
    class DummyTokenizer:
        def __call__(self, text, return_tensors): return {}
        def decode(self, *args, **kwargs): return "Model not loaded. Cannot process LLM queries."
        eos_token_id = 0
    class DummyModel:
        def generate(self, *args, **kwargs): return torch.tensor([[0]])
        def eval(self): pass
        device = "cpu"
    tokenizer = DummyTokenizer()
    finetuned_model = DummyModel()

# 5. Load JSON dataset
try:
    with open("test.json", "r", encoding="utf-8") as f:
        dataset_qa = json.load(f)
    logger.info(f"Loaded {len(dataset_qa)} items from test.json")
except FileNotFoundError:
    logger.error("Error: test.json not found. Please make sure the JSON file is in the correct directory.")
    dataset_qa = []
except json.JSONDecodeError:
    logger.error("Error: Could not decode test.json. Check for valid JSON format.")
    dataset_qa = []

# 6. Utility: Extract text from uploaded documents
def extract_text_from_file(doc):
    ext = os.path.splitext(doc.name)[1].lower()
    try:
        if ext == ".txt":
            with open(doc.name, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            text = ""
            with pdfplumber.open(doc.name) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            return text
        elif ext == ".docx":
            docx_file = docx.Document(doc.name)
            return "\n".join(para.text for para in docx_file.paragraphs)
        elif ext == ".csv":
            with open(doc.name, "r", encoding="utf-8") as f:
                return "\n".join(" ".join(row) for row in csv.reader(f))
        elif ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(doc.name)
            return pytesseract.image_to_string(image)
    except Exception as e:
        logger.error(f"Error reading file {doc.name}: {e}")
        return f"Error reading file: {e}"
    return ""

# 7. Utility: Search in dataset
def find_dataset_item(user_input):
    user_input_lower = user_input.strip().lower()
    instructions = [item["instruction"].strip().lower() for item in dataset_qa]
    for idx, instr in enumerate(instructions):
        if instr == user_input_lower:
            return dataset_qa[idx]
    for item in dataset_qa:
        if user_input_lower in item["instruction"].lower() or user_input_lower in item["input"].lower():
            return item
    close_matches = difflib.get_close_matches(user_input_lower, instructions, n=1, cutoff=0.7)
    if close_matches:
        idx = instructions.index(close_matches[0])
        return dataset_qa[idx]
    return None

def parse_fbs_hba1c(input_str):
    fbs = None
    hba1c = None
    fbs_match = re.search(r"fbs:\s*([\d.]+)(?:\s*mg/dL)?", input_str, re.IGNORECASE)
    hba1c_match = re.search(r"hba1c:\s*([\d.]+)(?:%)?", input_str, re.IGNORECASE)
    if fbs_match:
        try:
            fbs = float(fbs_match.group(1))
        except ValueError:
            pass
    if hba1c_match:
        try:
            hba1c = float(hba1c_match.group(1))
        except ValueError:
            pass
    return fbs, hba1c

# 8. Unified Chat Function
def chat(user_input, history, doc):
    history = history or []
    doc_text = None
    for item in history:
        if item.get("role") == "memory" and "doc_text" in item:
            doc_text = item["doc_text"]
            break

    if doc is not None and not doc_text:
        doc_text = extract_text_from_file(doc)
        if doc_text:
            history.append({"role": "memory", "doc_text": doc_text})
        else:
            history.append({"role": "memory", "doc_text": "Error: Could not extract text from document."})
            return [msg for msg in history if msg["role"] in ("user", "assistant")], history

    if (
        ("fbs:" in user_input.lower() and "hba1c:" in user_input.lower()) or
        (re.search(r"\bmg/dl\b", user_input.lower()) and re.search(r"%", user_input.lower())) or
        (re.search(r"hba1c\s*[:=\-]?[\d.]+%", user_input.lower()) and re.search(r"[\d.]+\s*mg/dl", user_input.lower()))
    ):
        user_fbs, user_hba1c = parse_fbs_hba1c(user_input)
        if user_fbs is None:
            fbs_match = re.search(r"([\d.]+)\s*mg/dl", user_input.lower())
            if fbs_match:
                try:
                    user_fbs = float(fbs_match.group(1))
                except ValueError:
                    user_fbs = None
        if user_hba1c is None:
            hba1c_match = re.search(r"hba1c\s*[:=\-]?\s*([\d.]+)%", user_input.lower())
            if not hba1c_match:
                hba1c_match = re.search(r"([\d.]+)%", user_input.lower())
            if hba1c_match:
                try:
                    user_hba1c = float(hba1c_match.group(1))
                except ValueError:
                    user_hba1c = None

        if user_fbs is not None and user_hba1c is not None:
            instr_types = [
                "what is the underwriting interpretation and decision for fbs < 100 mg/dl?",
                "what is the underwriting interpretation and decision?",
                "what is the underwriting interpretation?",
                "what does underwriting consider for mildly elevated fasting blood sugar (fbs = 100–110 mg/dl)?",
                "how does underwriting assess moderately elevated fbs levels (111–125 mg/dl)?",
                "how is underwriting impacted when fbs is in the diabetic range (≥126 mg/dl)?",
                "how does underwriting evaluate an applicant with known diabetes or hba1c > 7.0%?",
                "what is the underwriting guidance for clients with borderline sugar levels?",
                "why is an hba1c test required if the client’s fbs is only 112 mg/dl?",
                "what is the underwriting decision and table rating?",
                "what is the underwriting decision?"
            ]
            user_instr = user_input.strip().lower()
            wants_decision = "decision" in user_instr
            wants_interpret = "interpretation" in user_instr or any(x in user_instr for x in ["consider", "assess", "impacted", "evaluate", "guidance", "why is an hba1c test required"])
            reply_parts = []
            interpretation = None
            interpretation_priority = -1
            if wants_interpret:
                def matches_range(ds_input, fbs, hba1c):
                    ds_input = ds_input.lower()
                    if "hba1c >" in ds_input:
                        val = float(re.findall(r"hba1c\s*>\s*([\d.]+)", ds_input)[0])
                        if hba1c > val:
                            return 3
                    if "known diabetes" in ds_input:
                        if hba1c > 7.0 or fbs >= 126:
                            return 3
                    if "fbs ≥" in ds_input or "fbs >=" in ds_input:
                        val = float(re.findall(r"fbs\s*[≥>=]+\s*([\d.]+)", ds_input)[0])
                        if fbs >= val:
                            return 2
                    if "fbs =" in ds_input and "–" in ds_input:
                        vals = re.findall(r"fbs\s*=\s*([\d.]+)[–-]([\d.]+)", ds_input)
                        if vals:
                            low, high = map(float, vals[0])
                            if low <= fbs <= high:
                                return 1
                    if "borderline" in ds_input:
                        if (100 <= fbs <= 110) or (5.7 <= hba1c <= 6.4):
                            return 1
                    if "fbs <" in ds_input:
                        val = float(re.findall(r"fbs\s*<\s*([\d.]+)", ds_input)[0])
                        if fbs < val:
                            return 0
                    if "fbs is only" in ds_input:
                        val = float(re.findall(r"fbs is only ([\d.]+)", ds_input)[0])
                        if abs(fbs - val) < 2:
                            return 1
                    return -1
                for item in dataset_qa:
                    if any(kw in item["instruction"].strip().lower() for kw in ["interpretation", "consider", "assess", "impacted", "evaluate", "guidance", "why is an hba1c test required"]):
                        ds_input = item["input"]
                        priority = matches_range(ds_input, user_fbs, user_hba1c)
                        if priority > interpretation_priority:
                            interpretation_priority = priority
                            interpretation = item["output"]
                if interpretation:
                    reply_parts.append(f"Interpretation: {interpretation}")
            decision = None
            if wants_decision:
                min_dist = float('inf')
                best_item = None
                for item in dataset_qa:
                    if item["instruction"].strip().lower() == "what is the underwriting decision and table rating?":
                        ds_fbs, ds_hba1c = parse_fbs_hba1c(item["input"])
                        if ds_fbs is not None and ds_hba1c is not None:
                            dist = ((user_fbs - ds_fbs)**2 + (user_hba1c - ds_hba1c)**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                best_item = item
                if best_item and min_dist < 5.0:
                    output = best_item["output"]
                    if isinstance(output, dict):
                        for k, v in output.items():
                            reply_parts.append(f"{k.capitalize().replace('_', ' ')}: {v}")
                    else:
                        reply_parts.append(f"Decision: {output}")
            if not reply_parts:
                interpretation = None
                interpretation_priority = -1
                def matches_range(ds_input, fbs, hba1c):
                    ds_input = ds_input.lower()
                    if "hba1c >" in ds_input:
                        val = float(re.findall(r"hba1c\s*>\s*([\d.]+)", ds_input)[0])
                        if hba1c > val:
                            return 3
                    if "known diabetes" in ds_input:
                        if hba1c > 7.0 or fbs >= 126:
                            return 3
                    if "fbs ≥" in ds_input or "fbs >=" in ds_input:
                        val = float(re.findall(r"fbs\s*[≥>=]+\s*([\d.]+)", ds_input)[0])
                        if fbs >= val:
                            return 2
                    if "fbs =" in ds_input and "–" in ds_input:
                        vals = re.findall(r"fbs\s*=\s*([\d.]+)[–-]([\d.]+)", ds_input)
                        if vals:
                            low, high = map(float, vals[0])
                            if low <= fbs <= high:
                                return 1
                    if "borderline" in ds_input:
                        if (100 <= fbs <= 110) or (5.7 <= hba1c <= 6.4):
                            return 1
                    if "fbs <" in ds_input:
                        val = float(re.findall(r"fbs\s*<\s*([\d.]+)", ds_input)[0])
                        if fbs < val:
                            return 0
                    if "fbs is only" in ds_input:
                        val = float(re.findall(r"fbs is only ([\d.]+)", ds_input)[0])
                        if abs(fbs - val) < 2:
                            return 1
                    return -1
                for item in dataset_qa:
                    if any(kw in item["instruction"].strip().lower() for kw in ["interpretation", "consider", "assess", "impacted", "evaluate", "guidance", "why is an hba1c test required"]):
                        ds_input = item["input"]
                        priority = matches_range(ds_input, user_fbs, user_hba1c)
                        if priority > interpretation_priority:
                            interpretation_priority = priority
                            interpretation = item["output"]
                if interpretation:
                    reply_parts.append(f"Interpretation: {interpretation}")
                min_dist = float('inf')
                best_item = None
                for item in dataset_qa:
                    if item["instruction"].strip().lower() == "what is the underwriting decision and table rating?":
                        ds_fbs, ds_hba1c = parse_fbs_hba1c(item["input"])
                        if ds_fbs is not None and ds_hba1c is not None:
                            dist = ((user_fbs - ds_fbs)**2 + (user_hba1c - ds_hba1c)**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                best_item = item
                if best_item and min_dist < 5.0:
                    output = best_item["output"]
                    if isinstance(output, dict):
                        for k, v in output.items():
                            reply_parts.append(f"{k.capitalize().replace('_', ' ')}: {v}")
                    else:
                        reply_parts.append(f"Decision: {output}")
            if reply_parts:
                reply = "\n".join(reply_parts)
            else:
                reply = "No close matching underwriting rule found for the given FBS and HbA1c values in the dataset. Please provide valid inputs."
        else:
            reply = "Could not extract valid FBS and HbA1c values from your input. Please provide them in the format 'FBS: XX mg/dL, HbA1c: YY%'."

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        return [msg for msg in history if msg["role"] in ("user", "assistant")], history

    # If user asks for underwriting decision for the document, try to extract FBS/HbA1c from doc_text and use dataset logic
    if (("underwriting decision" in user_input.lower() or "table rating" in user_input.lower()) and doc_text and 'error reading file' not in doc_text.lower()):
        # Try to extract FBS and HbA1c from the document text
        fbs_match = re.search(r"fbs[\s:=-]*([\d]+[\.,]?[\d]*)", doc_text, re.IGNORECASE)
        hba1c_match = re.search(r"hba1c[\s:=-]*([\d]+[\.,]?[\d]*)", doc_text, re.IGNORECASE)
        fbs_val = float(fbs_match.group(1).replace(",", ".")) if fbs_match else None
        hba1c_val = float(hba1c_match.group(1).replace(",", ".")) if hba1c_match else None
        if fbs_val is not None or hba1c_val is not None:
            # Use dataset logic for decision/table rating, even if only one value is present
            min_dist = float('inf')
            best_item = None
            for item in dataset_qa:
                if item["instruction"].strip().lower() == "what is the underwriting decision and table rating?":
                    ds_fbs, ds_hba1c = parse_fbs_hba1c(item["input"])
                    # Only compare on available values
                    dist = 0
                    count = 0
                    if fbs_val is not None and ds_fbs is not None:
                        dist += (fbs_val - ds_fbs) ** 2
                        count += 1
                    if hba1c_val is not None and ds_hba1c is not None:
                        dist += (hba1c_val - ds_hba1c) ** 2
                        count += 1
                    if count > 0:
                        dist = dist ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_item = item
            if best_item and min_dist < 5.0:
                output = best_item["output"]
                if isinstance(output, dict):
                    reply = "\n".join([f"{k.capitalize().replace('_', ' ')}: {v}" for k, v in output.items()])
                else:
                    reply = str(output)
            else:
                reply = "No close matching underwriting rule found for the available FBS or HbA1c value(s) in the document."
        else:
            reply = "The document does not contain FBS or HbA1c values required for an underwriting decision."
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        return [msg for msg in history if msg["role"] in ("user", "assistant")], history

    elif doc_text and "error reading file" not in doc_text.lower():
        sub_questions = re.split(r'\band\b|[,;]', user_input, flags=re.IGNORECASE)
        sub_questions = [q.strip() for q in sub_questions if q.strip()]
        answers = []

        if not model_loaded:
            reply = "Cannot perform document-based QA as the LLM model failed to load."
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            return [msg for msg in history if msg["role"] in ("user", "assistant")], history

        explanation_keywords = ["why", "explain", "how", "reason", "justification", "clarify"]
        for sq in sub_questions:
            if 'result' in sq.lower() or 'all results' in sq.lower():
                value_keywords = ["hba1c", "fbs", "glucose", "eag", "sugar"]
                found = []
                for keyword in value_keywords:
                    pattern = re.compile(rf"{keyword}[\s:=-]*([\d]+(?:[.,]?[\d]*))(?:\s*(?:mg/dL|%)?)", re.IGNORECASE)
                    matches = pattern.findall(doc_text)
                    for m in set(matches):
                        found.append(f"{keyword.upper()}: {m.replace(',', '.')}" )
                if found:
                    reply_sub = ", ".join(found)
                    answers.append(reply_sub)
                    continue
            is_explanation_question = any(kw in sq.lower() for kw in explanation_keywords)
            # Try to answer explanation questions from dataset first
            dataset_explanation = None
            if is_explanation_question:
                # Try to extract FBS/HbA1c from doc_text if available
                fbs_val = None
                hba1c_val = None
                if doc_text and 'error reading file' not in doc_text.lower():
                    fbs_match = re.search(r"fbs[\s:=-]*([\d]+[\.,]?[\d]*)", doc_text, re.IGNORECASE)
                    hba1c_match = re.search(r"hba1c[\s:=-]*([\d]+[\.,]?[\d]*)", doc_text, re.IGNORECASE)
                    fbs_val = float(fbs_match.group(1).replace(",", ".")) if fbs_match else None
                    hba1c_val = float(hba1c_match.group(1).replace(",", ".")) if hba1c_match else None
                # Find closest match in dataset for explanation questions
                min_dist = float('inf')
                best_item = None
                for item in dataset_qa:
                    if any(kw in item["instruction"].lower() for kw in explanation_keywords):
                        ds_fbs, ds_hba1c = parse_fbs_hba1c(item["input"])
                        dist = 0
                        count = 0
                        if fbs_val is not None and ds_fbs is not None:
                            dist += (fbs_val - ds_fbs) ** 2
                            count += 1
                        if hba1c_val is not None and ds_hba1c is not None:
                            dist += (hba1c_val - ds_hba1c) ** 2
                            count += 1
                        if count > 0:
                            dist = dist ** 0.5
                            if dist < min_dist:
                                min_dist = dist
                                best_item = item
                if best_item and min_dist < 10.0:
                    dataset_explanation = best_item["output"]
            if dataset_explanation:
                reply_sub = dataset_explanation
            else:
                if doc_text and 'error reading file' not in doc_text.lower():
                    if is_explanation_question:
                        prompt = f"""### Instruction:\nYou are an underwriting analyst. The user wants to understand the reasoning behind a previous underwriting decision. Analyze the following medical document and provide a clear explanation for **why** the underwriting decision (e.g., Rated, Declined, Approved) would be made.\n\n### Document:\n{doc_text[:4000]}\n\n### Input:\n{sq}\n\n### Output:"""
                    else:
                        prompt = f"""### Instruction:\nYou are an underwriting assistant. Analyze the following medical document and answer the user's question accurately using the provided data (e.g., HbA1c, FBS).\n\n### Document:\n{doc_text[:4000]}\n\n### Input:\n{sq}\n\n### Output:"""
                else:
                    if is_explanation_question:
                        prompt = f"""### Instruction:\nYou are an underwriting analyst. The user wants to understand the reason behind a specific underwriting outcome. Based on standard guidelines, explain **why** an applicant might receive a decision like Rated, Declined, or Approved given their FBS and HbA1c levels.\n\n### Input:\n{sq}\n\n### Output:"""
                    else:
                        prompt = f"""### Instruction:\nYou are an underwriting assistant. Answer the user's question based only on underwriting guidelines.\n\n### Input:\n{sq}\n\n### Output:"""
                inputs = tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
                with torch.no_grad():
                    output = finetuned_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.7,
                        top_p=0.9,                     
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                reply_sub = decoded.split("### Output:")[-1].strip() if "### Output:" in decoded else decoded.strip()
            answers.append(reply_sub)
        reply = " ".join(answers)

    else:
        item = find_dataset_item(user_input)
        if item:
            output = item["output"]
            reply = json.dumps(output, ensure_ascii=False) if isinstance(output, (dict, list)) else str(output)
        else:
            reply = "I couldn't find relevant information for your question. Could you please rephrase it or provide more details?"

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return [msg for msg in history if msg["role"] in ("user", "assistant")], history

# === AUDIO FEATURE HELPERS ===
recognizer = sr.Recognizer()
audio_interface = pyaudio.PyAudio()
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
audio_frames = []
recording = False
record_thread = None

def start_recording():
    global recording, audio_frames, record_thread
    recording = True
    audio_frames = []
    def record_audio():
        global recording, audio_frames
        try:
            stream = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            while recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_frames.append(data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            pass
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()
    return "🎤 Recording... Click 'Stop Recording' when done."

def stop_recording():
    global recording, audio_frames, record_thread
    recording = False
    time.sleep(0.5)
    if record_thread:
        record_thread.join(timeout=1)
    if not audio_frames:
        return "❌ No audio recorded.", ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        wf = wave.open(temp_audio.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        try:
            with sr.AudioFile(temp_audio.name) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language='en-US')
                return f"✅ Transcribed: {text}", text
        except sr.UnknownValueError:
            return "❌ Could not understand audio.", ""
        except sr.RequestError as e:
            return f"❌ Speech recognition error: {e}", ""
        finally:
            try:
                os.unlink(temp_audio.name)
            except Exception:
                pass

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            return temp_file.name
    except Exception:
        return None

# 9. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot — Unified Intelligence (Underwriting + Document QA)")
    chatbot = gr.Chatbot(label="Chat", type="messages")
    msg = gr.Textbox(placeholder="Ask something...", label="Your Question")
    doc = gr.File(label="Upload any document (PDF, DOCX, TXT, CSV, Image)")
    state = gr.State([])

    # === AUDIO FEATURE UI ===
    with gr.Row():
        with gr.Column():
            record_btn = gr.Button("🎤 Record Voice", variant="secondary")
            stop_btn = gr.Button("⏹️ Stop Recording", variant="stop")
            record_status = gr.Textbox(label="Recording Status", interactive=False)
            transcribed_text = gr.Textbox(label="Transcribed Text", interactive=True)
        with gr.Column():
            play_audio_btn = gr.Button("🔊 Play Last Response", variant="secondary")
            audio_output = gr.Audio(label="AI Response Audio", type="filepath")

    # === AUDIO FEATURE EVENTS ===
    record_btn.click(start_recording, inputs=None, outputs=record_status)
    stop_btn.click(stop_recording, inputs=None, outputs=[record_status, transcribed_text])
    stop_btn.click(lambda status, text: text, inputs=[record_status, transcribed_text], outputs=msg)

    def play_last_audio(history):
        if not history:
            return None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                audio_file = text_to_speech(msg["content"])
                return audio_file
        return None
    play_audio_btn.click(play_last_audio, inputs=state, outputs=audio_output)

    # === EXISTING CHAT EVENTS ===
    msg.submit(chat, [msg, state, doc], [chatbot, state])
    msg.submit(lambda: "", None, msg)

demo.launch(server_name="127.0.0.1")