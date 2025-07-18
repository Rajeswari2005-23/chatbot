from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openlm-research/open_llama_3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
