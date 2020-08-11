import re 
import os
import pandas as pd
import json
import numpy as np
import pandas as pd
import pickle

PATH_TO_TOKENIZED_SPEECHES = '../congressional_speeches_tokenized'
PATH_TO_PARSED_SPEECHES = '../congressional_speeches_parsed'
PATH_TO_FRAME_OUTPUT = '../saved_frames'
PATH_TO_LOGFILES = '../logfiles'
LETTER_PATTERN = re.compile('[A-Za-z]+')
INVALID_POS = ['PUNCT', 'X', 'SYM', 'SPACE']
MIN_SPEECH = 43
MAX_SPEECH = 114

CPU_USAGE_THRESHOLD = 70
MEM_USAGE_THRESHOLD = 70
SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD = 10

################################################################
# Anchor terms
################################################################
PERSON_ANCHORS = ['man', 'woman', 'person', 'people', 'individual', 'population', 'group']
IMMIGRANT_ANCHORS = ['immigrant', 'emigrant', 'migrant', 'alien', 'foreigner', 'refugee']
CHINESE_ANCHORS = ['chinese', 'chinaman', 'chinamen', 'coolie', 'cooly', 'mongol', 'mongolian']
AMERICAN_ANCHORS = ['american']
GROUP_ANCHORS = ['american', 'german', 'british', 'irish', 'italian', 'european', 'russian', 
                 'chinese', 'japanese', 'mongolian', 'asian',
                 'hispanic', 'mexican', 'latino']

################################################################
# Data loading functions
################################################################
def get_tokenized_speech_fn(num):
    fn = os.path.join(PATH_TO_TOKENIZED_SPEECHES, 'speeches_%03d.jsonlist' % num)
    return fn

def get_parsed_speech_fn(num):
    fn = os.path.join(PATH_TO_PARSED_SPEECHES, 'speeches_%03d_parsed.csv' % num)
    return fn

def get_frames_for_speech_fn(num):
    fn = os.path.join(PATH_TO_FRAME_OUTPUT, 'speeches_%03d_frames.csv' % num)
    return fn

def load_tokenized_speech(num):
    fn = get_tokenized_speech_fn(num)
    print('Loading speech from %s' % fn)
    assert os.path.isfile(fn)
    f = open(fn, 'r')
    content = f.readlines()
    json_list = [json.loads(line[:-1]) for line in content[1:]]  # leave out header line, truncate \n on each line
    print('Found %d sections in speech.' % len(content))
    return json_list

def load_parsed_speech(num, only_valid=True):
    fn = get_parsed_speech_fn(num)
    print('Loading speech from %s' % fn)
    df = pd.read_csv(fn)
    if only_valid:
        orig_len = len(df)
        df = df[~df['pos'].isin(INVALID_POS)]
        print('Dropped %d/%d tokens with POS in %s' % (orig_len - len(df), len(df), INVALID_POS))
    return df

def load_frames_for_speech(num):
    fn = os.path.join(PATH_TO_FRAME_OUTPUT, 'speeches_%03d_frames.csv' % num)
    print('Loading frames from %s' % fn)
    df = pd.read_csv(fn)
    bad_columns = [col for col in df.columns if col.endswith('.1')]
    df = df.drop(columns=bad_columns)  # extra columns that got saved by accident
    return df