from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

p = "The black horse"
s = "down the ramp."
p = "World hunger"
s = "are unfair."
p = "The boy had"
s = "a crash on a girl"
prompt = f"first half:{p} second half:{s}. If this sentence is incorrect, provide a corrected version of the second half such that the sentence requires the minimal number of edits. Provide one sentence only.\nCorrected version:"
messages = [
    {"role": "system", "content": "You are a teacher."},
                {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=10,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    # temperature=0.7,
    # top_p=0.9,
    # repetition_penalty=1.1,
    # early_stopping=True,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

# Get model output (for debugging)
# out = model(tokenizer(prompt, return_tensors="pt").input_ids.to(model.device))
# 
# out = model(tokenizer(p, return_tensors="pt").input_ids.to(model.device))
# Compute attention mask and input ids
attention_mask = tokenizer(p, return_tensors="pt").attention_mask.to(model.device)
input_ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)
# Complete the prompt
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=10,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    # early_stopping=True,
)
text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\n\n")
print(text)


# Batch variation
print("\n\nBATCH\n\n")
prompts = [
   f"first half:{p} second half:{s}. If this sentence is incorrect, provide a corrected " 
   "version of the second half such that the sentence requires the minimal number of edits. "
   "Provide one sentence only.\nCorrected version:"
    for p, s in [(p, s), ("The cat sat", "at the mat."), ("The sun", " are shining.")]
]
tokenizer.padding_side = "left"
texts = tokenizer.apply_chat_template(
    [[
        {"role": "system", "content": "You are a teacher."},
        {"role": "user", "content": p}
    ] for p in prompts],
    tokenize=False,
    add_generation_prompt=True,
    # padding_side="left"
)
model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
attention_mask = model_inputs.attention_mask
input_ids = model_inputs.input_ids
# Complete the prompt
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=10,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    # temperature=0.7,
    # top_p=0.9,
    # repetition_penalty=1.1,
    # early_stopping=True,
)
generated_ids = [
    output_ids[len(input_ids_):] for input_ids_, output_ids in zip(input_ids
, generated_ids)
]
responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("\n\n")
for r in responses:
    print(r)            