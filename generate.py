from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

# Fix padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.eval()

prompt = input("Enter your prompt: ")

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True
)

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        do_sample=True,              # VERY IMPORTANT
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Text:\n")
print(generated_text)
