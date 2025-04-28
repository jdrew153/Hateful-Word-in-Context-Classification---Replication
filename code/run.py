from embeddings import get_embedding_file
from tenfold_eval import evaluate
from dinu_eval import dinu_evaluate
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import experiments

def run(data_path, id_column, label_column, label_encoder,
                   embedding_dir, predictions_dir, logs_path,
                   models, model_layers, clf, embedding_types,
                   splitby, params=dict(), random_split_seed=12):
    
    logs = []
    for model in models:
        model_name = model.rsplit('/',1)[1] if '/' in model else model
        model_name = model_name.rsplit('.',1)[0] if '.' in model_name else model_name
        for embedding_type in embedding_types:
            experiment_description = f'\n{clf} / {model_name} / {embedding_type} embeddings / {model_layers} layer(s) / split by {splitby}\n'.upper()
            logs.append(experiment_description)
            logs.append(f"Hyperparameters: {params}")
            print(experiment_description)
            # default embeddings are token embeddings of example usages
            embedding_path = get_embedding_file(data_path, id_column, embedding_dir, embedding_type, model, model_layers)
            predictions_path = predictions_dir + f'{clf}-{model_name}-{embedding_type}-{model_layers}-splitby{splitby}.csv'
            if 'dinu' in data_path:
                experiment_logs = dinu_evaluate(data_path, label_column, embedding_path, predictions_path, clf, params, random_split_seed)
            else:
                experiment_logs = evaluate(data_path, id_column, label_column, label_encoder, embedding_path, 
                                            predictions_path, clf, params, random_split_seed, splitby) 
            logs.extend(experiment_logs)
            #print(experiment_logs)
    
    with open(logs_path, 'w') as outfile:
        for string in logs:
            outfile.write(string+'\n')

# here goes nothing...
dataset = "C:\\Users\\jdrew\\OneDrive\\Desktop\\CompSci\\NLP\\Final\\data\\HateWiC\\HateWiC_preprocessed.csv"

df = pd.read_csv(dataset)
le = LabelEncoder()
le.fit(df['label'])


dirs_needed = ['predictions', 'logs', 'embeddings']

for dir in dirs_needed:
    if not os.path.isdir(dir):
        os.makedirs(dir)



# for bert_exp in experiments.experiment_run_params_BERT:
#     run(
#         data_path=dataset,
#         id_column="id",
#         label_column="label",
#         label_encoder=le,
#         embedding_dir="embeddings\\",
#         predictions_dir="predictions\\",
#         logs_path=f'logs\\{bert_exp['logs_path']}',
#         models=bert_exp['models'],
#         model_layers="last",
#         clf="mlp", ## either mlp or dimproj
#         embedding_types=bert_exp['embedding_types'],
#         splitby=bert_exp['splitby'] # -> split by term for OOV, and split by example for random
#     )


# for bert_exp in experiments.remaining_hateBERT_exps:
#     run(
#         data_path=dataset,
#         id_column="id",
#         label_column="label",
#         label_encoder=le,
#         embedding_dir="embeddings\\",
#         predictions_dir="predictions\\",
#         logs_path=f'logs\\{bert_exp['logs_path']}',
#         models=bert_exp['models'],
#         model_layers="last",
#         clf="mlp", ## either mlp or dimproj
#         embedding_types=bert_exp['embedding_types'],
#         splitby=bert_exp['splitby'] # -> split by term for OOV, and split by example for random
#     )


for bert_exp in experiments.experiments_DINU:
    df = pd.read_csv(bert_exp['data_path'])
    le = LabelEncoder()
    le.fit(df['label'])
    run(
        data_path=bert_exp['data_path'],
        id_column="id",
        label_column="label",
        label_encoder=le,
        embedding_dir="embeddings\\",
        predictions_dir="predictions\\",
        logs_path=f'logs\\{bert_exp['logs_path']}',
        models=bert_exp['models'],
        model_layers="last",
        clf="mlp", ## either mlp or dimproj
        embedding_types=bert_exp['embedding_types'],
        splitby=bert_exp['splitby'] # -> split by term for OOV, and split by example for random
    )
