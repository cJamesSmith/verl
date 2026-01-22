import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/jfzn/yhc/models/Qwen/HF-Qwen3-8B-sft"
tokenizer_path = "/mnt/jfzn/yhc/models/Qwen/HF-Qwen3-8B-sft"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = (
    AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    .cuda()
    .to(torch.bfloat16)
)


system_prompt = "你是一个严谨、耐心的计算机算法老师，回答要有清晰思路和代码示例。"
user_prompt = "你是一个应届生，给我写一段求最长公共子序列的DP算法。"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
    {
        "content": "Question:\nLet $a,$ $b,$ $c$ be real numbers such that\n\\[|ax^2 + bx + c| \\le 1\\]for all $0 \\le x \\le 1.$  Find the largest possible value of $|a| + |b| + |c|.$\n",
        "role": "user",
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("===== INPUT TEXT =====")
print(text)

model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=5120,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
)


output_ids = generated_ids[0][model_inputs.input_ids.shape[-1] :]

content = tokenizer.decode(output_ids, skip_special_tokens=False)

print("===== OUTPUT =====")
print(content)
