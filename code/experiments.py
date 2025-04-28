## list of parameters for each run of the experiment
experiment_run_params_BERT = [
    ## 1. BERT Random - WiC 
    {
        'logs_path' : 'BERT_WiC_Random.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['example'],
        'splitby' : 'example' # -> split by term for OOV, and split by example for random
    },
    ## 2. BERT OoV - WiC 
    {
        'logs_path' : 'BERT_WiC_OoV.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['example'],
        'splitby' : 'term'
    },
    ## 3. BERT Random - Def 
    {
        'logs_path' : 'BERT_Def_Random.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['definition'],
        'splitby' : 'example'
    },
     ## 4. BERT OoV - Def 
    {
        'logs_path' : 'BERT_Def_OoV.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['definition'],
        'splitby' : 'term'
    },
    ## 5. BERT Random - T5Def 
    {
        'logs_path' : 'BERT_T5Def_Random.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['generated_definition'],
        'splitby' : 'example'
    },
     ## 6. BERT OoV - T5Def
    {
        'logs_path' : 'BERT_T5Def_OoV.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['generated_definition'],
        'splitby' : 'term'
    },
    ## 7. BERT Random - WiC + Def 
    {
        'logs_path' : 'BERT_WiC_Def_Random.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['example', 'definition'],
        'splitby' : 'example'
    },
     ## 8. BERT OoV - WiC + Def
    {
        'logs_path' : 'BERT_WiC_Def_OoV.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['example', 'definition'],
        'splitby' : 'term'
    },

    ## 8. BERT Random - WiC + T5Def 
    {
        'logs_path' : 'BERT_WiC_T5Def_Random.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['example', 'generated_definition'],
        'splitby' : 'example'
    },
     ## 10. BERT OoV - WiC + T5Def
    {
        'logs_path' : 'BERT_WiC_T5Def_OoV.log',
        'models' : ['google-bert/bert-base-uncased'],
        'embedding_types' : ['example', 'generated_definition'],
        'splitby' : 'term'
    },
]


experiment_run_params_HateBERT = [
    ## 1. hateBERT Random - WiC 
    {
        'logs_path'     : 'hateBERT_WiC_Random.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['example'],
        'splitby'       : 'example'  # split by example for random
    },
    ## 2. hateBERT OoV - WiC 
    {
        'logs_path'     : 'hateBERT_WiC_OoV.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['example'],
        'splitby'       : 'term'     # split by term for OoV
    },
    ## 3. hateBERT Random - Def 
    {
        'logs_path'     : 'hateBERT_Def_Random.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['definition'],
        'splitby'       : 'example'
    },
    ## 4. hateBERT OoV - Def 
    {
        'logs_path'     : 'hateBERT_Def_OoV.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['definition'],
        'splitby'       : 'term'
    },
    ## 5. hateBERT Random - T5Def 
    {
        'logs_path'     : 'hateBERT_T5Def_Random.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['generated_definition'],
        'splitby'       : 'example'
    },
    ## 6. hateBERT OoV - T5Def
    {
        'logs_path'     : 'hateBERT_T5Def_OoV.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['generated_definition'],
        'splitby'       : 'term'
    },
    ## 7. hateBERT Random - WiC + Def 
    {
        'logs_path'     : 'hateBERT_WiC_Def_Random.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['example', 'definition'],
        'splitby'       : 'example'
    },
    ## 8. hateBERT OoV - WiC + Def
    {
        'logs_path'     : 'hateBERT_WiC_Def_OoV.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['example', 'definition'],
        'splitby'       : 'term'
    },
    ## 9. hateBERT Random - WiC + T5Def 
    {
        'logs_path'     : 'hateBERT_WiC_T5Def_Random.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['example', 'generated_definition'],
        'splitby'       : 'example'
    },
    ## 10. hateBERT OoV - WiC + T5Def
    {
        'logs_path'     : 'hateBERT_WiC_T5Def_OoV.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['example', 'generated_definition'],
        'splitby'       : 'term'
    },
]

# remaining_hateBERT_exps = [
#      ## 3. hateBERT Random - Def 
#     {
#         'logs_path'     : 'hateBERT_Def_Random.log',
#         'models'        : ['GroNLP/hateBERT'],
#         'embedding_types': ['definition'],
#         'splitby'       : 'example'
#     },
#     ## 4. hateBERT OoV - Def 
#     {
#         'logs_path'     : 'hateBERT_Def_OoV.log',
#         'models'        : ['GroNLP/hateBERT'],
#         'embedding_types': ['definition'],
#         'splitby'       : 'term'
#     },
#       ## 7. hateBERT Random - WiC + Def 
#     {
#         'logs_path'     : 'hateBERT_WiC_Def_Random.log',
#         'models'        : ['GroNLP/hateBERT'],
#         'embedding_types': ['example', 'definition'],
#         'splitby'       : 'example'
#     },
#     ## 8. hateBERT OoV - WiC + Def
#     {
#         'logs_path'     : 'hateBERT_WiC_Def_OoV.log',
#         'models'        : ['GroNLP/hateBERT'],
#         'embedding_types': ['example', 'definition'],
#         'splitby'       : 'term'
#     },
# ]

remaining_hateBERT_exps = [
     ## 3. hateBERT Random - Def 
    {
        'logs_path'     : 'hateBERT_Def_Random.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['definition'],
        'splitby'       : 'example'
    },
    ## 4. hateBERT OoV - Def 
    {
        'logs_path'     : 'hateBERT_Def_OoV.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['definition'],
        'splitby'       : 'term'
    },
      ## 7. hateBERT Random - WiC + Def 
    {
        'logs_path'     : 'hateBERT_WiC_Def_Random.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['example', 'definition'],
        'splitby'       : 'example'
    },
    ## 8. hateBERT OoV - WiC + Def
    {
        'logs_path'     : 'hateBERT_WiC_Def_OoV.log',
        'models'        : ['GroNLP/hateBERT'],
        'embedding_types': ['example', 'definition'],
        'splitby'       : 'term'
    },
]

experiments_DINU = [
   {
        'data_path' : 'C:\\Users\\jdrew\\OneDrive\\Desktop\\CompSci\\NLP\\Final\\data\\DINU\\DINU1.csv',
        'logs_path' : 'DINU1_HateBert_WiC_Random.log',
        'models' : ['GroNLP/hateBERT'],
        'embedding_types' : ['example'],
        'splitby' : 'example' # -> split by term for OOV, and split by example for random
    },
]