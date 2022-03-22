import os, glob, json
import pickle
import pandas as pd
pd.set_option('display.max_colwidth', None)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import re
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
import random, math
from tqdm import tqdm
import platform
import argparse
from cleantext import clean

clean_str = lambda s: clean(s,
                            fix_unicode=True,               # fix various unicode errors
                            to_ascii=True,                  # transliterate to closest ASCII representation
                            lower=True,                     # lowercase text
                            no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
                            no_urls=True,                  # replace all URLs with a special token
                            no_emails=True,                # replace all email addresses with a special token
                            no_phone_numbers=True,         # replace all phone numbers with a special token
                            no_numbers=False,               # replace all numbers with a special token
                            no_digits=False,                # replace all digits with a special token
                            no_currency_symbols=True,      # replace all currency symbols with a special token
                            no_punct=False,                 # fully remove punctuation
                            replace_with_url="<URL>",
                            replace_with_email="<EMAIL>",
                            replace_with_phone_number="<PHONE>",
                            replace_with_number="<NUMBER>",
                            replace_with_digit="0",
                            replace_with_currency_symbol="<CUR>",
                            lang="en"
                            )
    
def main(fname_to_texts_to_process,text_col_name,output_dirname = 'combined_stanza_output_with_ner_embed_removed'):
    
    df = pickle.load(open(fname_to_texts_to_process,'rb'))
    print(f"Read in df to process from {fname_to_texts_to_process} with dimensions {df.shape}.")

    for _,row in tqdm(df.iterrows()):
        savename = os.path.join(output_dirname,f"{row['id']}.json")
        try:
            doc = nlp(row[text_col_name])
            jsonlist = {'parse': [[{
                                        'id': word.id,
                                        'text': word.text,
                                        'lemma': word.lemma,
                                        'upos': word.upos,
                                        'xpos': word.xpos,
                                        'feats': word.feats,
                                        'head': word.head,
                                        'deprel': word.deprel,
                                        'start_char': word.start_char,
                                        'end_char': word.end_char
                                    } 
                        for word in sent.words]
                        for sent in doc.sentences],
                        'ents': [{
                                    "text": ent.text,
                                    "type": ent.type,
                                    "start_char": ent.start_char,
                                    "end_char": ent.end_char
                                }
                                for ent in doc.ents]}
            with open(savename,'w') as f:
                json.dump(jsonlist,f)
        except AssertionError:
            pass
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_to_machine', action='store_true',
                        help='whether to write parsed output to /jag disk')
    parser.add_argument('--fname_to_texts_to_process', type=str, default="",
                        help='fname to specific texts to process (as dataframe)')
    parser.add_argument('--text_column_name', type=str, default="text",
                        help='column corresponding to text field in df to process')
    parser.add_argument('--processed_dir', type=str,
                        default='combined_stanza_output_with_ner_embed_removed',
                        help='path to where already processed jsons are located')
    args = parser.parse_args()
    fname_to_texts_to_process = args.fname_to_texts_to_process
    text_column_name = args.text_column_name
    processed_dir = args.processed_dir
    
    print('Loading model...')
    import stanza
    stanza.download('en') # download English model
    nlp = stanza.Pipeline('en', use_gpu= True)
    
    if args.write_to_machine:
        # print machine name
        machine_name = platform.node().split(".")[0]
        print(f"Output will be written to disk of {machine_name}.")

        # print available scratch directories
        print("Available scratch directories:\n")
        print(" ".join(os.listdir(f"/{machine_name}")))
        scr_dir = os.listdir(f"/{machine_name}")[0]

        DISK_IO_DIR = "/{}/{}/yiweil".format(machine_name,scr_dir)
        output_dir = os.path.join(DISK_IO_DIR,processed_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = processed_dir
    print('Output dir:',output_dir)

    print('Processing specific texts from df...')
    main(fname_to_texts_to_process,text_column_name,output_dirname=output_dir)
    print('Done!')
