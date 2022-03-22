import os, glob, json
import pickle
import pandas as pd
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

from urllib.parse import urlparse

def get_url_domain(url):
    url = re.split(']|\)',url)[0]
    domain = urlparse(url).netloc
    #re_split = re.split('.com|.org|.gov|.edu|.net|.co|.ca|.int|.us|.be|.de|.cn|.uk|.tw|.cc', domain.split('www.')[-1])
    domain = domain.split('www.')[-1]
    short_domain = sorted(domain.split('.'), key=lambda x: len(x), reverse=True)[0]
    return (domain, short_domain)

def get_urls(text):
    urls = re.findall(r'(https?://\S+)', text)
    urls = [url.replace('[dot]','.').replace('[at]','@').replace('.m.','.').replace('.i.','.') for url in urls]
    return urls

def flatten(l):
    return [item for sublist in l for item in sublist]

def encode_surrogates(s):
    """Encodes a string s containing emoji and other special chars for saving."""
    return s.encode('utf-16', 'surrogatepass').decode('utf-16')

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
    
# bot/moderator-filtering utils
known_bots = pickle.load(open('../1_data_collection/reddit/bots/bots.pkl','rb'))
with open('../1_data_collection/reddit/known_cmv_mods.txt','r') as f:
    known_cmv_mods = f.read().splitlines()
bot_mod_set = known_bots | set(known_cmv_mods)
bot_mod_set.add('TotesMessenger')

def written_by_bot(corp,utt_idx,bot_mod_set):
    return corp.get_utterance(utt_idx).speaker.id in bot_mod_set

def filter_bots_mods(corp,bot_mod_set=bot_mod_set,filter_bot_suffixes=True,do_filter=False):
    
    if filter_bot_suffixes:
        print("Filtering moderators and speakers with usernames ending in '-bot'...")
        speaker_ids = corp.get_speaker_ids()
        speaker_names = [corp.get_speaker(s_id).id for s_id in speaker_ids]
        potential_bots = [x for x in speaker_names if type(x) == str and x.lower().endswith('bot') 
                          and 'robot' not in x.lower()
                          and 'bottle' not in x.lower() and 'bottom' not in x.lower()]
        bot_mod_set |= set(potential_bots)
        del speaker_ids
        
    for utt in corp.iter_utterances():
        utt.meta['written_by_bot'] = written_by_bot(corp,utt.id,bot_mod_set)
        
    utt_ids_to_keep = set([utt.id for utt in corp.iter_utterances()
                          if not utt.meta['written_by_bot']])
    print("Found {} non-deleted/removed utterances not written by a bot or moderator.".format(
                len(utt_ids_to_keep)))
    
    if do_filter:
        return corp.filter_utterances_by(lambda utt: utt.id in utt_ids_to_keep)
    else:
        return corp

with open('generic_comment_regex_patterns.txt','r') as f:
    generic_regex_pats = f.read().splitlines()
GENERIC_REGEX_PATS = [re.compile(pat, re.IGNORECASE) for pat in generic_regex_pats]

def is_generic(utt_text,compiled_pats):
    for pat in compiled_pats:
        res = re.search(pat, utt_text)
        if res:
            return True
    return False

def filter_generic_comments(corp,generic_regex_pats,save=True,
                            fname='non_generic_utt_ids',cache=True,do_filter=False):
    
    if cache:
        generic_dict = pickle.load(open(fname,'rb')) 
        for utt in tqdm(corp.iter_utterances()):
            utt.meta['is_generic'] = generic_dict[utt.id]
    else:
        generic_dict = {}
        for utt in tqdm(corp.iter_utterances()):
            bool_ = is_generic(utt.text,compiled_pats=generic_regex_pats)
            utt.meta['is_generic'] = bool_
            generic_dict[utt.id] = bool_
                
        if save:
            pickle.dump(generic_dict,open('{}.pkl'.format(fname),'wb'))
        
    print("Filtering to {} non-generic comments.".format(
                    Counter(generic_dict.values())[False]))

    if do_filter:
        return corp.filter_utterances_by(lambda utt: (generic_dict[utt.id]==False) or (utt.id[:2] == 't3'))
    else:
        return corp

def cleanup(utt_text):
    lines = [line for line in utt_text.splitlines()
             if not line.lstrip().startswith("&gt;")
             and not line.lstrip().startswith("____")
             and not line.lstrip().startswith(">")
             and "edit" not in " ".join(line.lower().split()[:2])
            ]
    return "\n".join(lines)

def is_empty_utt(utt):
    return len(utt.text.replace('-','').replace('/','').strip()) == 0
    
from convokit import Corpus#, Speaker, download
AVAILABLE_CORP = ['full_cmv','matched_cmv_subsample','matched_cc_cmv_subsample','gen_reddit']
corp_name_to_filepath = {
                         'full_cmv': '/u/scr/yiweil/my-cmv-corpus_1-1-2010_to_09-24-2021',
                         'matched_cmv_subsample': '/u/scr/yiweil/subsampled_cmv_corpus',
                         'matched_cc_cmv_subsample':'/u/scr/yiweil/for_Esin/subsampled_cmv_corpus',
                         'gen_reddit':'/u/scr/yiweil/datasets/convokit/reddit-corpus-small',
                         'switchboard':'/sailhome/yiweil/.convokit/downloads/switchboard-corpus'
                        }

STANZA_DIR='/nlp/u/yiweil/combined_stanza_output_with_ner_embed_removed'

# Loads stanza-processed json object if utterance text is non-empty and not a single '/' or '-'
import json_stream
def get_stanza_jsonstream(utt,
    stanza_dir=STANZA_DIR):
    #if len(utt.text) > 0 and utt.text != '-' and utt.text != '/':
    return json_stream.load(open(os.path.join(STANZA_DIR,f'{utt.id}.json'),'r'), persistent=True)

def get_stanza_parse(utt):
    json_obj = get_stanza_jsonstream(utt)
    if json_obj is not None:
        return json_obj['parse']
    else:
        return []
    
def get_stanza_lemmas(utt, adx_only=False, exclude_punc=True):
    json_obj = get_stanza_jsonstream(utt)
    if json_obj is not None:
        parse = json_obj['parse']
        if exclude_punc:
            lemmas =  [(x['lemma'],x['upos']) for sent in parse for x in sent
                       if x['upos'] != 'PUNCT']
        else:
            lemmas = [(x['lemma'],x['upos']) for sent in parse for x in sent]
        if adx_only:
            return [x[0] for x in lemmas if x[1] in {'ADJ','ADV'}]
        else:
            return [x[0] for x in lemmas]
    else:
        return []
    
deltalog = pd.read_csv(os.path.join('/u/scr/yiweil/Green-American-Lexicon/1_data_collection/reddit',
                                    'deltalog.tsv'),
                      sep='\t',header=0)
delta_awarded_utt_ids = set(deltalog['awarded_utt_id'].values)
delta_counts = deltalog['awarded_utt_id'].value_counts()
utt_id2num_deltas = {id_: delta_counts[id_] for id_ in delta_counts.index.values}

# Update meta to include whether an utterance was awarded a delta or not
def is_awarded_delta(utterance_id):
    return utterance_id in delta_awarded_utt_ids

def num_deltas(utterance_id):
    if utterance_id in delta_awarded_utt_ids:
        return utt_id2num_deltas[utterance_id]
    else:
        return 0
    
DELTAS = ['Δ','&#8710;','∆','!delta']
DELTA_REGEX = r'{}'.format('|'.join(DELTAS))
print(f'Using {DELTA_REGEX} as regex for comments awarding delta.')

def awarding_delta(utt, delta_regex=DELTA_REGEX):
    return (utt.id[:2]=='t1') and (utt.speaker.id != 'DeltaBot') and (len(re.findall(DELTA_REGEX, utt.text)) > 0)
    
manual_checked = pd.read_csv('is_cc_related_gold_manual_checked.tsv',sep='\t')
manual_checked['is_true'] = manual_checked['is_true'].apply(
    lambda x: {'f':'false','m':'meta','d':'doom','r':'renewables/energy','v':'vegetarianism',
              'p':'overpopulation','g':'government/politics','c':'conservation','e':'economy',
              'l':'lifestyle','foo':'food (non-vegetarianism)','a':'adaptation',
              'mor':'ethics/morality','gm':'government/politics'}[x]
                if not pd.isnull(x) else 'other')
manual_checked['label'] = manual_checked['label'].apply(lambda x: {'s':'skeptic','st':'indifferent'}[x] 
                                                       if x in {'s','st'} else 'other')
gold_df = manual_checked.loc[manual_checked['is_true']!='false']
climate_convo_idxs_gold = set(gold_df['id'].values)
print(f'Found {len(climate_convo_idxs_gold)} conversations which are manually checked to be true climate convos.')

c_id2other_cc_label = dict(zip(gold_df['id'].values,gold_df['is_true'].values))

def is_outcome_leaf(utt):
    """Returns whether a comment represents an outcome node."""
    return (utt.id[:2] == 't1') and (utt.meta['outcome'] != -1)

def is_OP_utt(corp,utt):
    if utt.speaker.id == None:
        return False
    return (utt.speaker.id == corp.get_utterance(utt.conversation_id).speaker.id)

def get_path_to_outcome_utt(corp, outcome_utt, ignore_OP=True, ignore_removed=True):
    """Returns the path of utterance IDs (excluding parent post) all the way to the outcome_utt (incl.)."""
    assert is_outcome_leaf(outcome_utt)
    if ignore_OP:
        path_to_outcome_utt = [u for u in outcome_utt.meta['path_to_outcome']
                               if not is_OP_utt(corp, corp.get_utterance(u))]
    else:
        path_to_outcome_utt = outcome_utt.meta['path_to_outcome']
    if ignore_removed:
        path_to_outcome_utt = [u for u in path_to_outcome_utt
                               if not corp.get_utterance(u).meta['is_removed']]
    return path_to_outcome_utt

def get_rel_timestamp_subtree(corp,outcome_utt):
    """
    Returns the time elapsed in seconds between the first comment of a sub-tree and the OG post.
    """
    first_utt_in_subtree = corp.get_utterance(outcome_utt.meta['path_to_outcome'][0])
    return first_utt_in_subtree.timestamp - corp.get_utterance(outcome_utt.conversation_id).timestamp

def get_abs_timestamp_subtree(corp,outcome_utt):
    """
    Returns the absolute timestamp of the first comment of a sub-tree.
    """
    first_utt_in_subtree = corp.get_utterance(outcome_utt.meta['path_to_outcome'][0])
    return first_utt_in_subtree.timestamp

def get_context(corp,utt_id,sent_ix,word_ix,window_before=3,window_after=20):
    sent = get_stanza_json(corp.get_utterance(utt_id))['parse'][sent_ix]
    context = ' '.join([sent[word]['text'] for word in range(max(0,word_ix-window_before),
                                                    min(len(sent),word_ix+window_after))])
    return context

def get_utt(utt_id,corp):
    return corp.get_utterance(utt_id)

def is_valid_utt(utt):
    """
    Returns whether an utterance meets the following criteria: 1) is not removed/deleted; 2) is not empty; 3) is not generic; 
    4) is not written by a bot/moderator.
    """
    return (not utt.meta['is_removed']) and (not is_empty_utt(utt)) and (not utt.meta['is_generic']) \
       and (not utt.meta['written_by_bot']) 

removed_outcome_ids = pickle.load(open('removed_subtrees.pkl','rb'))
def is_removed_subtree(subtree_outcome_utt,path_to_removed_subtrees=removed_outcome_ids):
    return subtree_outcome_utt.id in removed_outcome_ids

abbrev_dict = {
    3: 'K',
    6: 'M',
    9: 'B'}

def abbreviate_N(n):
    """Returns a str representation of an int n to abbreviate for plotting."""
    str_n = str(n)
    if len(str_n) > 9:
        ix_cutoff = len(str_n)-9
        rounded_tail = round(float(str_n[ix_cutoff:ix_cutoff+2])/100,1)
        return str_n[:ix_cutoff]+str(rounded_tail)[1:] + abbrev_dict[9]
    elif len(str_n) > 6:
        ix_cutoff = len(str_n)-6
        rounded_tail = round(float(str_n[ix_cutoff:ix_cutoff+2])/100,1)
        return str_n[:ix_cutoff]+str(rounded_tail)[1:] + abbrev_dict[6]
    elif len(str_n) > 3:
        ix_cutoff = len(str_n)-3
        rounded_tail = round(float(str_n[ix_cutoff:ix_cutoff+2])/100,1)
        return str_n[:ix_cutoff]+str(rounded_tail)[1:] + abbrev_dict[3]
    else:
        return str_n