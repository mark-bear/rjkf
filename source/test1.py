import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_prompt_from_txt(file_path):
    with open(file_path, 'r') as f:
        prompts = f.read().strip().replace("\n\u200b\u200b","")
    prompts=[_[2:-1] for _ in prompts.strip().split("TASK")[1:]]
    return prompts

def generare_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 生成代码（调整参数优化输出）
    generated_ids = model.generate(
        **inputs,
        max_length=4196,          # 调整生成文本的最大长度
        temperature=0.7,          # 控制随机性（0-1，值越小越确定）
        top_p=0.9,                # 核采样参数
        do_sample=True,           # 启用随机采样
        pad_token_id=tokenizer.eos_token_id
    )

    # 解码生成的代码
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../../codegen25",trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("../../codegen25",device_map="auto")
    prompts= get_prompt_from_txt("prompt_of_CodeGen.txt")
    output_dir="../output"
    for prompt in prompts:
        print(f"doing{prompts.index(prompt)+1}:{prompt}")
        with open(f"{output_dir}/output_{prompts.index(prompt)}.txt", "w") as f:
            start_time=time.time()
            output=generare_code(prompt)
            end_time=time.time()
            f.write(f"generate time:{end_time-start_time}\n")
            f.write(output)