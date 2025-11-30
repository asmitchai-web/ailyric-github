import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Initialize the tokenizer and model
# The model weights will be downloaded automatically the first time you run this
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

# 2. Define your input text (prompt)
prompt = "In a shocking finding, scientists discovered a herd of unicorns"

# 3. Encode the input text into token IDs
# 'pt' specifies PyTorch tensors; use 'tf' for TensorFlow
inputs = tokenizer.encode(prompt, return_tensors='pt')

# 4. Generate new text
# The 'generate' method handles the text generation process
outputs = model.generate(
    inputs, 
    max_length=100, 
    do_sample=True, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True
)

# 5. Decode the generated tokens back into human-readable text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 6. Print the output
print("---Generated Text---")
print(generated_text)
