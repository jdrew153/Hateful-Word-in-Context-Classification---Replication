from transformers import pipeline
import re
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

llama_model_name = 'meta-llama/Llama-2-7b-chat-hf'

'''
    Format prompt per paper specification
'''
def format_prompt(example, term):
    prompt = f'''
        ## Instruction:

        Given the following sentence that mentions
        a particular term, classify whether
        the meaning of that term expresses hate
        towards a person or group within that
        specific sentence. Respond with exactly
        one of the following corresponding
        labels without an explanation:
        “HATEFUL”
        “NOT HATEFUL”

        ### Input:
        Sentence: {example}
        Term: {term}

        ### Response:
'''
    return prompt


    
df = pd.read_csv("C:\\Users\\jdrew\\OneDrive\\Desktop\\CompSci\\NLP\\Final\\data\\HateWiC\\HateWiC_preprocessed.csv")

# load once
tokenizer = AutoTokenizer.from_pretrained(llama_model_name, padding_side='left')
model     = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    torch_dtype=torch.float16,    # half – precision
    device_map="auto",            # scatter across GPUs / device 0
)
model.eval()

tokenizer.pad_token_id       = tokenizer.eos_token_id
tokenizer.pad_token          = tokenizer.eos_token
model.config.pad_token_id    = tokenizer.eos_token_id

batch_size = 16
results = []

for start in range(0, len(df), batch_size):
    print(f'On :{start}')
   

    batch_df = df.iloc[start:start + batch_size].copy()
   
    prompts = [
        format_prompt(row['example'], row['term'])
        for _, row in batch_df.iterrows()
    ]
   
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    # 3) generate
    with torch.no_grad():
        out_ids = model.generate(
            **enc.to(model.device),
            max_new_tokens=10,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
   

    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
   
    responses = [
        text.split("### Response:")[1].strip().splitlines()[0]
        for text in decoded
    ]

    batch_df = batch_df.copy()
    batch_df['label'] = responses
    results.append(batch_df)

df_with_responses = pd.concat(results).sort_index()
# print(df_with_responses[['example','term','label']])

df_with_responses.to_csv("test_llama.csv", index=False)