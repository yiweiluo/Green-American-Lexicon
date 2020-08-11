import argparse
from ast import literal_eval
from constants_and_utils import *
from collections import Counter
import datetime
import math
import multiprocessing
from nltk.corpus import wordnet as wn
import numpy as np
import os
import pandas as pd
import pickle
import psutil
from psutil._common import bytes2human
import spacy
import subprocess
import time 

def parse_sections_with_spacy(sections):
    '''
    Parses sentences into tokens annotated for lemma, part-of-speech, dependency, and head.
    Returns a DataFrame with one row per token.
    '''
    section_ids = []
    sent_ids = []
    sents = []
    for section_id, section in enumerate(sections):
        for sent_id, sent in enumerate(section['sents']):
            section_ids.append(section_id)
            sent_ids.append(sent_id)
            sents.append(sent)
    print('Num sentences to parse:', len(sents))
    
    nlp = spacy.load("en_core_web_sm")
    tokens = []
    overall_sent_id = 0
    start_time = time.time()
    # sents is a list of strings, each string represents one sentence
    for doc in nlp.pipe(sents, disable=["ner"], n_threads=8):
        for tok in doc:
            tokens.append({
                'section_id':section_ids[overall_sent_id],
                'sent_id':sent_ids[overall_sent_id],
                'tok_id':tok.i,
                'text':tok.text,
                'lemma':tok.lemma_,
                'pos':tok.pos_,
                'tag':tok.tag_,
                'dep':tok.dep_,
                'head_id':tok.head.i,
                'head_lemma':tok.head.lemma_,
              }
            )
        overall_sent_id += 1
        if overall_sent_id % 10000 == 0:
            curr_time = time.time()
            print('Parsed %d sentences in %.3fs -> %.5fs per sentence' % 
                  (overall_sent_id, curr_time - start_time, (curr_time - start_time) / overall_sent_id))
    end_time = time.time()
    print('Finished parsing sentences in %.3fs -> %.5fs per sentence' % 
          (end_time - start_time, (end_time - start_time)/len(sents)))
    assert overall_sent_id == len(sents)
    results = pd.DataFrame(tokens)
    return results

def parse_and_save_speech(speech_num, out_fn):
    '''
    Parses a speech and saves the output at out_fn.
    '''
    print('Saving parsed tokens in %s' % out_fn)
    tokenized = load_tokenized_speech(speech_num)
    df = parse_sections_with_spacy(tokenized)
    df.to_csv(out_fn, index=False)
    return df

def get_all_tokens_within_n_steps(parsed_tokens_df, orig_anchors, n):
    '''
    Takes in the parsed tokens dataframe (output of parse_sections_with_spacy) and a subset of 
    the rows as the original anchors. 
    Returns the rows of all tokens that are within n-steps of the original anchors, with four 
    new columns appended:
    (1) first_anchor_id - the id of the source anchor on the path to this token,
    (2) first_anchor - the lemma of the source anchor on the path to this token,
    (3) node_str - all nodes (i.e., lemmas) on the path from the source anchor to this token,
    (4) edge_str - all edges (i.e., dependencies) on the path from the source anchor to this token
    
    For example, in "restricting Chinese immigration," the anchor is "Chinese", whose amod head is 
    "immigration", whose dobj head is "restricting" (lemma=restrict). So, this would return 
    first_anchor="chinese", node_str="chinese_people_restrict", edge_str="amod-head_dobj-head", 
    and first_anchor_id would be the id of the "Chinese" token.
    '''
    curr_anchors = orig_anchors.copy()
    curr_anchors['first_anchor_id'] = list(orig_anchors.index)
    curr_anchors['node_str'] = orig_anchors.lemma.values
    curr_anchors['edge_str'] = np.array(['' for i in range(len(orig_anchors))])
    all_rows = []
    for i in range(n):
        print('Getting %d-step tokens...' % (i+1))
        curr_anchors = _get_all_tokens_within_one_step(parsed_tokens_df, curr_anchors)
        curr_anchors['first_anchor'] = curr_anchors.node_str.map(lambda x:x.split('_')[0])
        all_rows.append(curr_anchors.copy())
        print()
    all_rows = pd.concat(all_rows)
    return all_rows

def _get_all_tokens_within_one_step(parsed_tokens_df, anchor_rows):
    '''
    Helper function to complete one iteration of get_all_tokens_within_n_steps. 
    Returns heads and dependents of current anchor rows.
    '''
    st = time.time()
    # get heads of anchor tokens
    nonroot_anchor_rows = anchor_rows[~(anchor_rows.dep == 'ROOT')]  # only want heads of non-roots (head of root is itself)
    anchor_head_lemmas = set(nonroot_anchor_rows.head_lemma.values)
    potential_head_rows = parsed_tokens_df[parsed_tokens_df['lemma'].isin(anchor_head_lemmas)].copy()
    potential_head_rows['complete_id'] = list(potential_head_rows.index)
    print('Found %d potential head tokens' % len(potential_head_rows))  # want to shorten to potential to speed up look-up time
    anchor_head_ids = list(zip(nonroot_anchor_rows.speech_id, nonroot_anchor_rows.section_id, nonroot_anchor_rows.sent_id, nonroot_anchor_rows.head_id))
    set_of_head_rows = potential_head_rows[potential_head_rows['complete_id'].isin(anchor_head_ids)]
    head_rows = set_of_head_rows.loc[anchor_head_ids]  # set and list are different bc one token could be head of multiple anchors
    print('Found %d head tokens of anchors; %d are unique' % (len(head_rows), len(set_of_head_rows)))
    head_rows['first_anchor_id'] = nonroot_anchor_rows.first_anchor_id.values
    head_rows['node_str'] = ['%s_%s' % (old_nodes, new_node) for old_nodes, new_node in zip(nonroot_anchor_rows.node_str, head_rows.lemma)]
    head_edge_strs = []
    for old_deps, new_dep in zip(nonroot_anchor_rows.edge_str, nonroot_anchor_rows.dep):
        if len(old_deps) > 0:
            head_edge_strs.append('%s_%s-head' % (old_deps, new_dep))
        else:
            head_edge_strs.append('%s-head' % new_dep)
    head_rows['edge_str'] = head_edge_strs
    head_rows.drop(columns=['complete_id'], inplace=True)
        
    # get dependents of anchor tokens
    anchor_lemmas = set(anchor_rows.lemma.values)
    potential_dependent_rows = parsed_tokens_df[parsed_tokens_df['head_lemma'].isin(anchor_lemmas)].copy()
    potential_dependent_rows = potential_dependent_rows[~(potential_dependent_rows.dep == 'ROOT')]  # only want non-root dependents (a root dependent would have itself as head)
    potential_dependent_rows['complete_head_id'] = list(zip(potential_dependent_rows.speech_id, potential_dependent_rows.section_id, potential_dependent_rows.sent_id, potential_dependent_rows.head_id))
    print('Found %d potential dependent tokens' % len(potential_dependent_rows))
    columns_to_update = zip(anchor_rows.first_anchor_id, anchor_rows.node_str, anchor_rows.edge_str)
    anchor_id_to_columns = dict(zip(anchor_rows.index, columns_to_update))
    dependent_rows = potential_dependent_rows[potential_dependent_rows['complete_head_id'].isin(anchor_id_to_columns.keys())]
    print('Found %d dependent tokens' % len(dependent_rows))  # no need to check num unique bc dependent can only have one head
    dependent_first_anchor_ids = []
    dependent_node_strs = []
    dependent_edge_strs = []
    for i, row in dependent_rows.iterrows():
        first_anchor_id, old_nodes, old_deps = anchor_id_to_columns[row['complete_head_id']]
        dependent_first_anchor_ids.append(first_anchor_id)        
        dependent_node_strs.append('%s_%s' % (old_nodes, row['lemma']))
        new_dep = row['dep']
        if len(old_deps) > 0:
            dependent_edge_strs.append('%s_%s-dep' % (old_deps, new_dep))
        else:
            dependent_edge_strs.append('%s-dep' % new_dep)
    dependent_rows['first_anchor_id'] = dependent_first_anchor_ids
    dependent_rows['node_str'] = dependent_node_strs
    dependent_rows['edge_str'] = dependent_edge_strs
    dependent_rows.drop(columns=['complete_head_id'], inplace=True)
    
    all_rows = pd.concat([head_rows, dependent_rows])
    is_cycle = all_rows.index == all_rows.first_anchor_id  
    print('Dropping %d cycles' % np.sum(is_cycle))  # don't want to keep paths that have cycled back to source anchor
    is_null = all_rows.lemma.isnull()
    print('Dropping %d null rows' % np.sum(is_null))  # sometimes .loc leads to row that was dropped, e.g., bc it was punctuation
    all_rows = all_rows[~(is_cycle | is_null)]
    print('Finished: found %d tokens one step from %d anchors [time = %.3f]' % 
          (len(all_rows), len(anchor_rows), time.time()-st))
    return all_rows

def get_all_wordnet_descendants(root):
    all_descendants = {root}
    curr_row = [root]
    next_row = []
    while len(curr_row) > 0:  # breadth first search
        for c in curr_row:
            hyps = c.hyponyms()
            next_row.extend(hyps)
        new_hyps = set(next_row) - all_descendants
        all_descendants = all_descendants.union(new_hyps)
        curr_row = list(new_hyps)
        next_row = []
    print('Found %d descendants of %s' % (len(all_descendants), root.name()))
    return all_descendants

def get_all_inhabitant_anchor_terms():
    inhabitant = wn.synset('inhabitant.n.01')
    all_inhabitants = get_all_wordnet_descendants(inhabitant)
    all_inhabitant_terms = set([w.lemmas()[0].name() for w in all_inhabitants])
    all_inhabitant_terms = set([w.lower() for w in all_inhabitant_terms if w == w.capitalize()])  # only want proper nouns
    print('Found %d unique inhabitant proper nouns' % len(all_inhabitant_terms))
    return all_inhabitant_terms
    
def compute_and_save_frames_for_speech(speech_num, out_fn, max_path_length=2):
    '''
    Extracts frames for a speech and saves the output at out_fn.
    '''
    print('Saving frames in %s' % out_fn)
    all_inhabitant_terms = get_all_inhabitant_anchor_terms()
    lowercase_anchors = PERSON_ANCHORS + IMMIGRANT_ANCHORS + list(all_inhabitant_terms)
    capitalized_anchors = [a.capitalize() for a in lowercase_anchors]
    anchor_terms = lowercase_anchors + capitalized_anchors
    print('Found %d anchors terms in total' % len(anchor_terms))                
            
    speech = load_parsed_speech(speech_num)
    speech['speech_id'] = speech_num
    speech = speech.set_index(['speech_id', 'section_id', 'sent_id', 'tok_id'], drop=False)
    orig_anchors = speech[speech['lemma'].isin(anchor_terms)]  # rows whose lemmas match anchor terms
    frames = get_all_tokens_within_n_steps(speech, orig_anchors, max_path_length)
    frames.to_csv(out_fn)
    
    # print anchor terms in order to highest to lowest frequency
    counter = Counter(frames.first_anchor.values)
    anchor_counts = []
    for a, cap_a in zip(lowercase_anchors, capitalized_anchors):
        count = 0
        if a in counter:
            count += counter[a]
        if cap_a in counter:
            count += counter[cap_a]
        anchor_counts.append(('%s/%s' % (a, cap_a), count))
    anchor_counts = sorted(anchor_counts, key=lambda x:x[1], reverse=True)
    for i, (anchor, count) in enumerate(anchor_counts):
        if count >= 10:
            print('%d. %s, n=%d (%.1f%%)' % (i+1, anchor, count, 100 * count / len(frames)))
        else:
            break

def simple_extract_frames(tokens, anchors, num_before, num_after):
    '''
    Simpler version of extracting frames: just take all words within fixed window of anchor terms,
    no dependency parsing needed. 
    Returns a DataFrame with one row per anchor instance.
    '''
    found_anchors = []
    found_before = []
    found_after = []
    for tokenized_sent in tokens:
        sent = [tok.lower() for tok in tokenized_sent if re.match(LETTER_PATTERN, tok)]  # ignore non-alpha tokens
        intersection = set(sent).intersection(set(anchors))
        for a in intersection:
            ind = sent.index(a)
            found_anchors.append(a)
            start = max(0, ind-num_before)
            found_before.append(sent[start:ind])
            end = min(ind+num_after+1, len(sent))
            found_after.append(sent[ind+1:end])
    df = pd.DataFrame({'anchor':found_anchors,
                       'words_before':found_before,
                       'words_after':found_after})
    return df

def produce_word_counts(df):
    '''
    Takes in simple frames dataframe (output of simple_extract_frames).
    Returns a Counter over all extracted frames (before and after).
    '''
    if type(df.iloc[0].words_before) == str:  # list was saved as string
        before_words = [w for wordlist in df.words_before.values for w in literal_eval(wordlist)]
        after_words = [w for wordlist in df.words_after.values for w in literal_eval(wordlist)]
    else:
        before_words = [w for wordlist in df.words_before.values for w in wordlist]
        after_words = [w for wordlist in df.words_after.values for w in wordlist]
    all_words = before_words + after_words
    counter = Counter(all_words)
    return counter

def compare_word_counts_with_log_odds(counter1, counter2, prior_counter=None, prior_weight=1000):
    '''
    Compares counts from two counters using log odds ratios and Dirichlet prior.
    '''
    left_words = set(counter1.keys())
    right_words = set(counter2.keys())
    all_words = sorted(left_words.union(right_words))  # get all words in either counter
    
    left_counts = np.array([counter1[w] if w in counter1 else 0 for w in all_words])
    right_counts = np.array([counter2[w] if w in counter2 else 0 for w in all_words])
    if prior_counter is None:
        print('Not passing in prior -> using +1 smoothing')
        prior_counts = np.zeros(len(all_words))
        left_counts = left_counts + 1
        right_counts = right_counts + 1
    else:
        # assumes Dirichlet priors for group 1 and group 2 are the same
        prior_counts = np.array([prior_counter[w] if w in prior_counter else 0 for w in all_words])
        print('Passed in prior -> has counts for %d/%d words' % (np.sum(prior_counts > 0), len(all_words)))
        prior_sum = np.sum(prior_counts)
        scaling = prior_weight / prior_sum
        prior_counts = prior_counts * scaling  # rescale so that prior is not extremely strong
    left_sum = np.sum(left_counts)
    right_sum = np.sum(right_counts)
    prior_sum = np.sum(prior_counts)
    print('num words in group 1 = %d, num words in group 2 = %d, prior weight = %.1f' % (left_sum, right_sum, prior_sum))
          
    # See Monroe et al., 2008
    left_log_odds = np.log(left_counts + prior_counts) - np.log(left_sum + prior_sum - left_counts - prior_counts)
    right_log_odds = np.log(right_counts + prior_counts) - np.log(right_sum + prior_sum - right_counts - prior_counts)
    log_odds_ratios = left_log_odds - right_log_odds  # Eq 16
    approx_variance = (1 / (left_counts + prior_counts)) + (1 / (right_counts + prior_counts))  # Eq 20
    approx_z_scores = log_odds_ratios / np.sqrt(approx_variance)  # Eq 22
    ratio_df = pd.DataFrame({'word':all_words, 
                             'left_count':left_counts, 
                             'left_freq': np.round(left_counts / left_sum, 3),
                             'right_count':right_counts,
                             'right_freq': np.round(right_counts / right_sum, 3),
                             'prior_count': np.round(prior_counts, 3),
                             'log_odds_ratio':log_odds_ratios,
                             'z_score':approx_z_scores})
    ratio_df = ratio_df.sort_values(by='log_odds_ratio', ascending=False)  # most associated with group 1 at top
    return ratio_df

def run_many_processes_in_parallel(process, overwrite):
    '''
    Use this function if you want to run the same process in parallel on many speeches.
    Currently implemented for two types of processes: parsing grammar and extracting frames.
    '''
    assert process in {'parse_grammar', 'extract_frames'}
    max_processes_for_user = int(multiprocessing.cpu_count() / 1.2)
    print("Maximum number of processes to run: %i" % max_processes_for_user)
    all_speeches = range(MIN_SPEECH, MAX_SPEECH+1)
    print('Total speeches to process: %d (%d to %d)' % (len(all_speeches), MIN_SPEECH, MAX_SPEECH))
    for i, speech in enumerate(all_speeches):  
        print("Starting job %d/%d" % (i+1, len(all_speeches)))
        if process == 'parse_grammar':
            out_fn = get_parsed_speech_fn(speech)
        else:
            out_fn = get_frames_for_speech_fn(speech)
        if os.path.isfile(out_fn) and (not overwrite):  # do not overwrite existing files unless specified 
            print('%s already exists, will not overwrite' % out_fn)
        else:
            if os.path.isfile(out_fn):
                print('WARNING: overwriting %s' % out_fn)
            t0 = time.time()
            # Check how many processes user is running.
            n_processes_running = int(subprocess.check_output('ps -fA | grep framing.py | wc -l', shell=True))
            print("Current processes running for user: %i" % n_processes_running)
            while n_processes_running > max_processes_for_user:
                print("Current processes are %i, above threshold of %i; waiting." % (n_processes_running, max_processes_for_user))
                time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
                n_processes_running = int(subprocess.check_output('ps -fA | grep framing.py | wc -l', shell=True))

            # don't swamp cluster. Check CPU usage.
            cpu_usage = psutil.cpu_percent()
            print("Current CPU usage: %2.3f%%" % cpu_usage)
            while cpu_usage > CPU_USAGE_THRESHOLD:
                print("Current CPU usage is %2.3f, above threshold of %2.3f; waiting." % (cpu_usage, CPU_USAGE_THRESHOLD))
                time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
                cpu_usage = psutil.cpu_percent()

            # Also check memory.
            available_memory_percentage = check_memory_usage()
            while available_memory_percentage < 100 - MEM_USAGE_THRESHOLD:
                print("Current memory usage is above threshold of %2.3f; waiting." % (MEM_USAGE_THRESHOLD))
                time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
                available_memory_percentage = check_memory_usage()

            # If we pass these checks, start a job.
            log_fn = os.path.join(PATH_TO_LOGFILES, '%s_speeches_%03d.out' % (process, speech))
            cmd = 'nohup python -u framing.py process_one_speech %s --speech %d --overwrite %s > %s 2>&1 &' % (process, speech, overwrite, log_fn)
            print("Command: %s" % cmd)
            os.system(cmd)
            time.sleep(3)
            print("Time between job submissions: %2.3f" % (time.time() - t0))

def check_memory_usage():
    virtual_memory = psutil.virtual_memory()
    total_memory = getattr(virtual_memory, 'total')
    available_memory = getattr(virtual_memory, 'available')
    free_memory = getattr(virtual_memory, 'free')
    available_memory_percentage = 100. * available_memory / total_memory
    # Free memory is the amount of memory which is currently not used for anything. This number should be small, because memory which is not used is simply wasted.
    # Available memory is the amount of memory which is available for allocation to a new process or to existing processes.
    print('Total memory: %s; free memory: %s; available memory %s; available memory %2.3f%%' % (
        bytes2human(total_memory),
        bytes2human(free_memory),
        bytes2human(available_memory),
        available_memory_percentage))
    return available_memory_percentage

if __name__ == '__main__':
    # sample command line arguments:
    # python framing.py process_all_speeches parse_grammar --overwrite False 
    # python framing.py process_one_speech extract_frames --speech 50 --overwrite True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('manager_or_worker_job', help='Is this the manager job or the worker job?',
        choices=['process_all_speeches', 'process_one_speech'])
    parser.add_argument('process', help='What process to run', choices=['parse_grammar', 'extract_frames'])
    parser.add_argument('--speech', help='The number of the speech to process', type=int)
    parser.add_argument('--overwrite', help='Whether to overwrite old files', type=bool, default=False)
    args = parser.parse_args()
    
    if args.manager_or_worker_job == 'process_all_speeches':
        assert args.speech is None
        run_many_processes_in_parallel(args.process, args.overwrite)
    else:
        assert args.speech is not None
        if args.process == 'parse_grammar':
            out_fn = get_parsed_speech_fn(args.speech)
            if (not os.path.isfile) or args.overwrite:  
                parse_and_save_speech(args.speech, out_fn)
        else:
            out_fn = get_frames_for_speech_fn(args.speech)
            if (not os.path.isfile) or args.overwrite:
                compute_and_save_frames_for_speech(args.speech, out_fn)
