import time
from functools import wraps

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def print_running_time(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = func(*args, **kwargs)

            elapsed_time = time.time() - start_time
            print(f'{name}: {elapsed_time:.1f}s')
            return result
        return wrapper
    return decorator


@print_running_time("chat")
def chat(txt, model, tokenizer, device='cuda:0'):
    model.generation_config = \
        GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat")
    messages = []
    messages.append({"role": "user", "content": txt})
    response = model.chat(tokenizer, messages)
    return response


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Chat",
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Chat",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    gt_jsonl_path = "data/qvhighlights/gt/highlight_val_release.jsonl"
    rephrased_jsonl_path = "data/qvhighlights/query/val.jsonl"

    with jsonlines.open(gt_jsonl_path, mode='r') as reader:
        gt_jsonl = list(reader)

    with jsonlines.open(rephrased_jsonl_path, mode='w') as writer:
        for line in tqdm(gt_jsonl):
            query = line["query"]

            txt = f"Original query: {query}. Task: Please rephrase the original query using different wording while maintaining the same intent and information. Provide five different rephrasings."

            response = chat(txt, model, tokenizer)
            
            response = response.strip().split('\n')
            rephrase_queries = [r.split('. ', 1)[1]
                                for r in response if response]

            line["rephrased_query"] = rephrase_queries
            writer.write(line)


main()
