# Linguistic framing of immigrants in Congressional speeches
## Method description
1. Design three sets of anchor terms: group 1, group 2, and the background group. For example, group 1 could be terms describing Chinese people ("Chinese"), group 2 could be terms describing American people ("American"), and the background group could be terms describing any type of person ("French", "Spanish", "Chinese", "Ethiopian", "American", etc).
2. For each set of anchors, find all tokens that satisfy some sort of POS / dependency path criteria relative to an anchor word. For example, you could collect all tokens X such that X satisfies one of the following dependency paths: (1) ANCHOR <-nsubj- X (e.g., "arrive" in "the Chinese are arrive"), (2) ANCHOR <-amod- PERSON_TERM <-nsubj- X (e.g., "arrive" in "the Chinese people arrived"). Or, you could simply collect all verbs within *n* dependency relations of an anchor word.
3. Compare the collected tokens for group 1 to the collected tokens for group 2 with log odds ratio, with the collected tokens for the background group acting as the informative Dirichlet prior. This will produce a weighted log odds ratio and a z-score for every token. The more positive the log odds ratio, the greater the inferred association with group 1; the more negative, the greater the inferred association with group 2. Some z-score cutoff might be helpful, e.g., the standard 1.96. For details, see [Monroe et. al, 2008](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf), section 3.

## Computational pipeline
1. Parse speeches for lemmas, part-of-speech (POS), and dependency structure: see `parse_sections_with_spacy`.
2. Collect all tokens within *n*-steps of anchor words: see `get_all_tokens_within_n_steps`. 
3. Compare subsets of tokens collected for different anchors using weighted log odds ratio: see `compare_word_counts_with_log_odds`

## Various details
1. This pipeline can generalized to run on any set of texts. However, alterations will need to be made throughout the pipeline: for example, `parse_sections_with_spacy` expects the input data to be formatted in a certain way, and `get_all_tokens_within_n_steps` expects each row to be indexed by the token's speech ID, section ID, sentence ID, and token ID, which is specific to the structure of the Congressional speeches.

2. Some processes need to be run once on each text, e.g., parsing or extracting frames. This can take a long time, so to speed this up, `framing.py` has code to kick off many jobs in parallel on one machine (see `run_many_processes_in_parallel`). This can be called with a command line argument. For example:
```
python framing.py process_all_speeches parse_grammar --overwrite False 
```
will parse all speeches but will not overwrite old parsed speech files if they exist. You can also use a command line argument to parse or extract frames for a single speech, e.g.,
```
python framing.py process_one_speech extract_frames --speech 50 --overwrite True 
```
which will extract 1-step and 2-step frames (the default n is 2) for speech 50 and overwrite any old extracted frames file for speech 50.
