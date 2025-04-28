from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import csv
import json
import random
import torch

assert torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'google/flan-t5-base'

tokenizer = AutoTokenizer.from_pretrained("ltg/flan-t5-definition-en-large")
model = AutoModelForSeq2SeqLM.from_pretrained("ltg/flan-t5-definition-en-large")

model.to(device)

def random_jsonl_row(path):
    chosen = None
    with open(path, newline='', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if random.randrange(i) == 0:
                chosen = obj

    return chosen

'''
    Prompt format: [Sentence] What is the definition of [Term]
'''
def debug_format_prompt(data):
    term = data['word']
    example = data['parts_of_speech'][0]['example_usages'][0]

    return f'{example} What is the definition of {term}', term

'''
    Expects that you formatted the prompt correctly before calling the function
'''
def generate_definition(prompt):
    ## print(f'Given prompt -> {prompt}')
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    generated_def = tokenizer.decode(outputs[0], skip_special_tokens=True)

    ## some defs have ' .' that I'd like to remove
    clean_def = generated_def.replace(' .', '').strip()
    return clean_def

