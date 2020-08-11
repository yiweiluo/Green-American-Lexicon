import os
import json
from optparse import OptionParser
from collections import Counter

import numpy as np
from scipy import stats

"""
Find words associated with a particular document attribute (the target value in the specified field)
Assumes that the text has been preprocessed, with tokens grouped by sentence

TODO:
- add tokenizer here directly
- allow for using cumulative vs point probabilities
"""


def main():
    usage = "%prog infile.jsonlist"
    parser = OptionParser(usage=usage)
    parser.add_option('--sents', type=str, default='tokens',
                      help='Field with tokens in sentences: default=%default')
    parser.add_option('--field', type=str, default='issue',
                      help='Field with target variable: default=%default')
    parser.add_option('--target', type=str, default='immigration',
                      help='Target value of field: default=%default')
    parser.add_option('--smoothing', type=float, default=0.1,
                      help='Amount of smoothing: default=%default')
    parser.add_option('--n-pos', type=int, default=50,
                      help='# pos to print: default=%default')
    parser.add_option('--n-neg', type=int, default=10,
                      help='# neg to print: default=%default')
    parser.add_option('--bigrams', action="store_true", default=False,
                      help="Use bigrams instead of unigrams: default=%default")
    parser.add_option('--no-lower', action="store_true", default=False,
                      help="Don't lower case: default=%default")
    parser.add_option('--cdf', action="store_true", default=False,
                      help="Use the CDF instead of the PMF: default=%default")

    (options, args) = parser.parse_args()

    infile = args[0]

    sents_field = options.sents
    field = options.field
    target = options.target
    smoothing = options.smoothing
    bigrams = options.bigrams
    lower = not options.no_lower

    print("Reading file")
    with open(infile) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]

    bg_counter = Counter()
    target_counter = Counter()

    print("Counting words")
    for line in lines:
        for sent in line[sents_field]:
            if lower:
                tokens = [t.lower() for t in sent]
            else:
                tokens = sent
            if bigrams:
                if len(tokens) > 1:
                    tokens = [tokens[i-1] + ' ' + tokens[i] for i in range(1, len(tokens))]
                else:
                    tokens = []
            if line[field] == target:
                target_counter.update(tokens)
            else:
                bg_counter.update(tokens)

    bg_total = sum(list(bg_counter.values()))
    target_total = sum(list(target_counter.values()))
    vocab = sorted(set(bg_counter).union(set(target_counter)))
    vocab_index = dict(zip(vocab, range(len(vocab))))

    print("{:d} words and {:d} tokens in background".format(len(bg_counter), bg_total))
    print("{:d} words and {:d} tokens in target".format(len(target_counter), target_total))
    print("Full vocab size = {:d}".format(len(vocab)))

    print("Converting to arrays")
    # convert background counts to a numpy array
    bg_freqs = np.zeros(len(vocab))
    for token, count in bg_counter.items():
        index = vocab_index[token]
        bg_freqs[index] = count

    # add smoothing
    bg_freqs += smoothing

    # normalize
    bg_freqs = bg_freqs / bg_freqs.sum()

    # convert target counts to a numpy array
    target_counts = np.zeros(len(vocab))
    for token, count in target_counter.items():
        index = vocab_index[token]
        target_counts[index] = count

    target_freqs = target_counts[:]
    target_freqs = target_freqs / target_freqs.sum()

    neglogprobs = -stats.binom.logpmf(target_counts, target_total, bg_freqs)
    neglogcdf = stats.binom.logcdf(target_counts, target_total, bg_freqs)

    if options.cdf:
        target = neglogcdf
    else:
        target = neglogprobs

    print("Common:")
    valid = np.array(target_freqs > bg_freqs)
    order = np.argsort(-target * valid)
    for i in range(options.n_pos):
        index = order[i]
        print(vocab[index], bg_freqs[index], target_counts[index] / target_total)

    print("\nRare:")
    valid = np.array(target_freqs < bg_freqs) * -1
    order = np.argsort(target * valid)
    for i in range(options.n_neg):
        index = order[i]
        print(vocab[index], bg_freqs[index], target_counts[index] / target_total)


if __name__ == '__main__':
    main()
