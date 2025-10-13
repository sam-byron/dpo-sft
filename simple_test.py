from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

p = "The black horse"
s = "down the ramp"
prompt = "Given the first half of a sentence denoted <sent1> first_half_sentence </sent1>"
"and a student's attempt to complete it denoted <student> student_attempt </student>, "
"provide a corrected version of the student's attempt in the format <corrected> corrected_text </corrected>.\n\n"
f"<sent1> {p} </sent1> <student> {s} </student>\n\n"
messages = [
    {"role": "system", "content": "You are a precise language teacher who cares about correct grammar, semantics, and style."},
                {"role": "user", "content": p}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)