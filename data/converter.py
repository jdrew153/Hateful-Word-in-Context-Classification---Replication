import json
import csv
import os
import re
import ast
from tqdm import tqdm
import pandas as pd
import random

from generate_t5_defs import generate_definition
import argparse

argp = argparse.ArgumentParser()
argp.add_argument("function", help='Choose from create_dataset, print_dataset, or annotate_dataset')
args = argp.parse_args()

'''
    To replicate HateWiC dataset need,
       - Example
       - Term
       - Definition
       - Annotations[] (Nh, Wh, Sh)
       - Binary labels[] (0 or 1s)
       - Majority label - (0 or 1)
       - Hate-Heterogeneous-Sense (True or False) [unique definitions for which example sentences exist in the dataset with both hateful and non-hateful majority annotations]

    Next Steps, just iterate through json file to get the term, but populate the HateWiC dataset with the example sentence with 
    corresponding definitions to their part of speech / word sense
'''

def clean_text(s: str) -> str:
    # keep A–Z, a–z, 0–9, whitespace, and %
    return re.sub(r'[^A-Za-z0-9%\s]', '', s).strip()

def helper_format_examples(data):
    '''
        data is coming in as {'word' : 'test', 'parts_of_speech': [{'part_of_speech' : 'Noun' , 'user_definition' : 'foo', 'example_usages : ['foot', 'hand']'}]}
        -> goal is to return [['example_usage', 'term', 'user_definition', 't5-def']] for each example usage for the respective part of speech
    '''
    ret = []

    term = data['word']
    ## print(f'\tProcessing term: {term}')

    parts_of_speech = data['parts_of_speech']

    for part_of_speech in parts_of_speech:
        ## definition has labels in (), remove for now.
        user_definition_w_labels = part_of_speech['user_definition']
        user_definition = re.sub(r'\s*\([^)]*\)', '', user_definition_w_labels).strip()

        example_usages = part_of_speech['example_usages']
       
        for example in example_usages:
            ## skip examples that are only one word, prob garbage
            if len(example) < 2:
                continue 

            ## Per spec in HateWiC paper
            prompt = f'{example} What is the definition of {term}'
            t5_def = generate_definition(prompt)

        
            ## in order specified above
            ret.append([example, term, user_definition, t5_def])
    
    return ret




def create_HateWiC_dataset(output_folder, json_filepath):
    HateWiC_csv_out_filename = "HateWiC_dataset.csv"
    HateWiC_csv_out_path = os.path.join(output_folder, HateWiC_csv_out_filename)

    with open(json_filepath, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(json_filepath, newline='', encoding='utf-8') as jf, \
        open(HateWiC_csv_out_path, 'w', newline='', encoding='utf-8') as cf:

        headers = ['Example', 'Term', 'Definition', 'T5-Definition']
        csv_writer = csv.writer(cf)

        csv_writer.writerow(headers)

        iteration_no = 0

        for line in tqdm(jf, total=total_lines, desc="Processing entries", unit="line", 
                         bar_format="{desc}: {n_fmt}/{total_fmt} [{bar}]"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                os.abort()
        
            formatted_examples = helper_format_examples(obj)
            csv_writer.writerows(formatted_examples)
            iteration_no += 1

def annotate_HateWiC_dataset(filepath):
    '''
        Add annotation column, write to an annotated version of the file.
    '''
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")


    df = pd.read_csv(filepath)

    if 'Annotations' in df.columns:
        df = df.drop(columns=['Annotations', 'label'])

    if 'label' not in df.columns:
        df['label'] = ''

    actual_annotation = None
    try:
        for idx in df.index:
            row = df.loc[idx]

            already_annotated = row['label']
            if pd.notna(already_annotated) and already_annotated != '':
                print(row['label'])
                print(f'Already annotated row {idx + 1}')
                continue
            
            print(f"\nRow {idx+1}/{len(df)}")
            print(f"Example   : {row['Example']}")
            print(f"Term      : {row['Term']}")
            print(f"Definition: {row['Definition']}")
            print(f"T5-Definition: {row['T5-Definition']}")
           
            ann = input("→ Enter a value 1 (Hate Speech) or 0 (Not Hatespeech): ").strip()

            actual_annotation = int(ann)

            df.at[idx, 'label'] = actual_annotation
    except KeyboardInterrupt:
        print("\n\nInterrupted! Attempting to save your progress...")
    
    finally:
        if actual_annotation is not None:    
            df.to_csv(filepath, index=False)
            print(f"\nWrote annotated CSV to {filepath}")
        else:
            print(f'No annotation was given, not saving file.')



def pre_process_HateWic_dataset(filepath, output_path):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")
    
    df = pd.read_csv(filepath)
    # insert id column
    new_df = df.copy()
    new_df.insert(0, 'id', df.index + 1)

    # Current labels - Example,Term,Definition,T5-Definition,Annotations,label
    #  - need headers like example, term, definition, profile_description, label
    new_df.rename(columns={
        'Example': 'example',
        'Term' : 'term',
        'Definition' : 'definition',
        'T5-Definition' : 'generated_definition'
    }, inplace=True)

    new_df = new_df.drop(columns=['Annotations'])
   
    new_df.dropna(inplace=True)

    new_df.to_csv(output_path, index=False)
    print(f'Wrote dataset prepped for model feeding to {output_path}')


def print_HateWiC_dataset(filepath):
    if not os.path.exists(filepath):
        return
    
    df = pd.read_csv(filepath)

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        print(f"Row {idx + 1}: {row_dict}")

def compare_llama_to_actual(truth_path, llama_path):
    df_truth = pd.read_csv(truth_path)
    df_pred = pd.read_csv(llama_path)

    df_truth = df_truth.copy()
    df_truth["label_str"] = df_truth["label"].map({True:"HATEFUL", False:"NOT HATEFUL"})

    df = pd.DataFrame({
        "true": df_truth["label_str"],
        "pred": df_pred["label"]   # already strings
    })

    acc = (df["true"] == df["pred"]).mean()
    print(f"Accuracy = {acc:.3%}")

output_folder = 'HateWiC'
data_folder = 'offensive'

hate_wic_filepath = os.path.join(output_folder,"HateWiC_dataset.csv")

llama_csv_path = 'C:\\Users\\jdrew\\OneDrive\\Desktop\\CompSci\\NLP\\Final\\test_llama.csv'

if args.function == 'create_dataset':
    create_HateWiC_dataset(output_folder, os.path.join(data_folder, 'offensive-parsed-page-content.json'))
elif args.function == 'print_dataset':
    print_HateWiC_dataset(hate_wic_filepath)
elif args.function == 'annotate_dataset':
    annotate_HateWiC_dataset(hate_wic_filepath)
elif args.function == 'preprocess_dataset':
    pre_process_HateWic_dataset(hate_wic_filepath, os.path.join(output_folder, 'HateWiC_preprocessed.csv'))
elif args.function == 'compare_llama':
    compare_llama_to_actual(os.path.join(output_folder, 'HateWiC_preprocessed.csv'), llama_csv_path)