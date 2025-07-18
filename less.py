import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
import argparse
import logging
import json
import difflib

# 1. Parse Arguments
parser = argparse.ArgumentParser(description="Run dual-model chatbot")
parser.add_argument("--model_dir", type=str, default="./outputs/final_model", help="Path to fine-tuned model dir")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Load Tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
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

# Load test.json for keyword-based responses
with open("test.json", "r", encoding="utf-8") as f:
    dataset_qa = json.load(f)

def find_dataset_item(user_input):
    user_input_lower = user_input.strip().lower()
    instructions = [item["instruction"].strip().lower() for item in dataset_qa]
    # Try exact match
    for idx, instr in enumerate(instructions):
        if instr == user_input_lower:
            return dataset_qa[idx]
    # Try substring match
    for idx, instr in enumerate(instructions):
        if instr in user_input_lower or user_input_lower in instr:
            return dataset_qa[idx]
    # Always return the closest fuzzy match (even if weak)
    close_matches = difflib.get_close_matches(user_input_lower, instructions, n=1, cutoff=0.0)
    if close_matches:
        idx = instructions.index(close_matches[0])
        return dataset_qa[idx]
    # Should never reach here if dataset is not empty
    return dataset_qa[0]

def chat(user_input, history):
    history = history or []
    item = find_dataset_item(user_input)
    # Use the user's question as the instruction, and optionally the matched input as context
    prompt = f"### Instruction:\n{user_input.strip()}\n\n### Input:{item['input'].strip()}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
    with torch.no_grad():
        output = finetuned_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "### Response:" in decoded:
        reply = decoded.split("### Response:")[-1].strip()
    else:
        reply = decoded.strip()

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return history, history

# 8. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖  Chatbot — Dual Mode (Fine-tuned + Base)")
    chatbot = gr.Chatbot(label="Chat", type="messages")
    msg = gr.Textbox(placeholder="Ask something...", label="Your Question")
    state = gr.State([])

    msg.submit(chat, [msg, state], [chatbot, state])
    msg.submit(lambda: "", None, msg)

demo.launch(server_name="127.0.0.1")
