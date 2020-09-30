import os
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import re
import math
import numpy as np
import pickle
import glob
import json
from json import JSONDecodeError
from collections import Counter, defaultdict
from urllib.parse import urlparse
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns
sns.set()
from tabulate import tabulate
from IPython.display import display, Image
import scipy.stats as stats
import statsmodels.formula.api as sm
from sklearn import metrics

from utils import clean_s, clean_url, corr_data

with open("config.json") as config_file:
    config = json.load(config_file)

class DataGetter:

    def __init__(self, data_fname):
        """
        :param data_fname: str filename of data within sub-directory
        """
        self.data_fname = data_fname
        self.out_dir = os.path.join(config['base_dir'],data_fname + '_out')
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.feats_dict = None
        self.data = None
        
    def get_dims(self):
        """Returns the dimensions of `self.data`."""
        return self.data.shape
    
    def get_data(self):
        """Returns the full dataframe `self.data`."""
        return self.data
    
    def get_feats_dict(self, ipython_disp=False):
        """Returns a dictionary with key feature types and val features."""
        if ipython_disp:
            display(pd.DataFrame.from_dict(self.feats_dict,orient='index').T)
        return self.feats_dict
    
    def get_out_dir(self):
        """Returns `self.out_dir."""
        return self.out_dir
    
    def load_data(self):
        """
        Initializes `self.data` to a pickled dataframe loaded from `data_dir/data_fname`. 
        Converts json-serialized fields in `self.data` into original list format.
        """
        data_path = os.path.join(config['base_dir'], config['data_dir'], self.data_fname)
        print('\nLoading data from {}...'.format(data_path))
        self.data = pd.read_pickle(data_path)
        self.data['title_lemmas'] = self.data['title_lemmas'].apply(lambda x: json.loads(x))
        self.data['body_lemmas'] = self.data['body_lemmas'].apply(lambda x: json.loads(x) if x else [])
        self.data['all_lemmas'] = self.data['title_lemmas'] + self.data['body_lemmas']
        self.data['all_text'] = self.data['title'].apply(lambda x: x + " ") + self.data['text']
        self.data = self.data.astype('object')
        print('Read data from with dimensions {}.'.format(self.get_dims()))
    
    def filter_data(self, exclude_bots=True, exclude_niche_subs=True, exclude_user_subs=True, 
        exclude_AMA=True, exclude_CMV=True, hi_prec_keyword_filter=True):
        """
        Filters data according to criteria given.
        
        :param exclude_bots: if True, excludes posts authored by bots (username ends in "bot" and/or 
            has authored extremely high number of comments)
        :param exclude_niche_subs: if True, excludes posts from niche or irrelevant subreddits
        :param exclude_user_subs: if True, excludes posts from mis-identified subreddit that is a user
        :param exclude_AMA: if True, excludes "ask me anything"-type posts
        :param exclude_CMV: if True, excludes "change my view"-type posts
        :param hi_prec_keyword_filter: if True, filters to posts whose text contains at least 
            2 distinct high precision keywords
        """
        print('\nFiltering data...')
        inputs_path = os.path.join(config['base_dir'], config['data_dir'])
        
        if exclude_bots:
            bots = pickle.load(open(inputs_path+'/bots.pkl','rb'))
            top_commenters = pickle.load(open(inputs_path+'/top_commenters.pkl','rb'))
            auths_to_exclude = set(bots) | set(top_commenters.index)
            print('Excluding posts from {} bot users...'.format(len(auths_to_exclude)))
            self.data = self.data.loc[~self.data.author.isin(auths_to_exclude)].copy()
            print('\tNew size of posts:',self.get_dims()[0])
            
        if exclude_niche_subs:
            niche_subs = set(open(inputs_path+'/niche_subs.txt','r').read().splitlines())
            print('Excluding posts from {} irrelevant subreddits...'.format(len(niche_subs)))
            self.data = self.data.loc[~self.data['subreddit'].isin(niche_subs)].copy()
            print('\tNew size of posts:',self.get_dims()[0])
            
        if exclude_user_subs:
            print("Excluding posts from a false subreddit that's actually a user...")
            self.data = self.data.loc[~self.data['subreddit'].apply(lambda x: x.startswith('u_') 
                                                                    if x else False)].copy()
            print('\tNew size of posts:',self.get_dims()[0])
            
        if exclude_AMA:
            
            def is_AMA(x):
                split_clean = clean_s(x).split()
                return 'AMA' in split_clean or 'IAmA' in split_clean
            
            print('Excluding AMA-type posts...')
            self.data = self.data.loc[~self.data['title'].apply(lambda x: is_AMA(x))].copy()
            print('\tNew size of posts:',self.get_dims()[0])
            
        if exclude_CMV:
            
            def is_CMV(x):
                split_clean = clean_s(x).split()
                return 'CMV' in split_clean or 'change my view' in x.lower()
            
            print('Excluding CMV-type posts...')
            self.data = self.data.loc[~self.data['title'].apply(lambda x: is_CMV(x))].copy()
            print('\tNew size of posts:',self.get_dims()[0])
            
        if hi_prec_keyword_filter:
            high_prec_words = set(open(inputs_path+'/high_precision_keywords.txt','r').read().splitlines())
            print('Filtering to posts that contain at least 2 distinct high precision keywords...')
            self.data = self.data.loc[self.data['all_lemmas'].apply(
                lambda x: len(set(x).intersection(high_prec_words)) > 1)].copy()
            print('\tNew size of posts:',self.get_dims()[0])
            
        self.data.reset_index(drop=True,inplace=True)

    def get_features(self, **params):
        """
        Adds new columns to `self.data` for all specified features.
        
        :param len: if True, adds log length (number of words) as feature
        :param url: if True, adds URL-related features (URLs contained in post; indicator variables for 
        most common URL domains; political leaning of URL domains contained)
        :param top_N_domains: int specifying the number of most common domains for the indicator variable 
        :param people: if True, adds indicator variables for mentions of people using people specified in 
        `most_common_people.txt`
        :param sentistrength: if True, adds SentiStrength positive and negative arousal scores as features
        :param vader: if True, adds VADER positive, negative, and compound sentiment scores as features
        :param textblob: if True, adds TextBlob sentiment and subjectivity scores as feature
        :param wiebe_subjectivity: if True, adds Wiebe et al. subjectivity clues score as feature
        :param NRC: if True, adds NRC VAD lexicon valence, arousal, dominance scores as features
        :param neg: if True, adds count of logical negations as feature
        :param nat_disaster: if True, adds count of natural disasters as feature
        :param economy: if True, adds count of economy-related words as feature
        :param emolex: if True, adds EmoIntensity lexicon emotion association scores as features
        :param morals: if True, adds Moral Foundations lexicon moral foundation association scores as features
        :param values: if True, adds Basic Human Values value association scores as features
        """
        print('\nAdding features...')
        # Set full path for reading inputs
        inputs_path = os.path.join(config['base_dir'], config['data_dir'])
        cache_prefix = os.path.join(config['base_dir'], self.out_dir, 'cached')
        os.mkdir(cache_prefix)
        
        # Set blacklist words to ignore when computing lexicon-based features 
        BLACKLIST_WORDS = set(open(inputs_path+'/blacklist_words.txt','r').read().splitlines())
        
        if params['length']:
            print('Adding length feature...')
            self.data['log_len'] = self.data['total_num_words'].apply(lambda x: math.log(x+0.1))
            print('\tDone!')
        
        if params['url']:
            print('Adding cached URL-related features...')
            
            # Read in data for URLs contained in each post
            urls_per_post_df = pd.read_csv(inputs_path+'/post_urls.tsv',sep='\t',header=0)
            urls_per_post_df = urls_per_post_df.drop_duplicates('post_id',keep='first')
            urls_per_post_df['post_urls'] = urls_per_post_df['post_urls'].apply(lambda x: json.loads(x))
            
            # Add URLs to the `url` API field as a new column in `self.data`; extract domain and regularize.
            self.data['urls'] = self.data['url'].apply(lambda x: [x] if x is not None else [])+\
                urls_per_post_df.loc[urls_per_post_df['post_id'].isin(self.data['id'])].reset_index()['post_urls']
            url_domains = self.data['urls'].apply(lambda x: [urlparse(item).netloc for item in x])
            self.data['clean_domains'] = url_domains.apply(lambda x: [clean_url(item) for item in x])
            
            # Add indicator variable for top N domains
            counted_domains = sorted(Counter([item for sublist in self.data['clean_domains'] 
                                              for item in sublist]).items(),
                                    key=lambda x: x[1],reverse=True)
            top_domains = [domain for domain,count in counted_domains if len(domain) > 0 and 'reddit' not in domain
                              and 'bit.ly' not in domain]
            for domain in top_domains[:params['top_N_domains']]:
                self.data['is_{}'.format(domain)] = self.data['clean_domains'].apply(
                    lambda x: 1 if domain in x else 0)
            
            # Add indicator variable for domain category (according to MediaBias)
            medbias_df = pd.read_csv(inputs_path+'/medbias_df.tsv',sep='\t',header=0)
            medbias_sets = {category: set([clean_url(url) for url in medbias_df[category].dropna()]) 
                for category in medbias_df.columns}
            for domain_cat in medbias_df.columns:
                self.data['is_{}'.format(domain_cat)] = self.data['clean_domains'].apply(
                    lambda x: 1 if len(set(x).intersection(medbias_sets[domain_cat])) > 0 else 0)
            print('\tDone!')
        
        if params['people']:
            print('Adding people-mention features...')
            most_common_people = set(open(inputs_path+'/most_common_people.txt','r').read().splitlines())
            for person in most_common_people:
                self.data['has_{}'.format(person)] = self.data['all_lemmas'].apply(lambda x: person in x)
            self.data['has_ocasiocortez'] = self.data['has_ocasio-cortez'].copy()
            del self.data['has_ocasio-cortez']
            print('\tDone!')
            
        if params['sentistrength']:
            cached_senti_path = os.path.join(cache_prefix,'sentistrength_scores_concatenated.tsv')
            if os.path.exists(cached_senti_path):
                print('Found cached feature file at {}. Adding cached SentiStrength features...'.format(cached_senti_path))
                sentistrength_df = pd.read_csv(cached_senti_path,sep='\t',header=0)
                senti_pos_scores = dict(zip(sentistrength_df['post_id'],sentistrength_df['sentistrength_pos_rating']))
                senti_neg_scores = dict(zip(sentistrength_df['post_id'],sentistrength_df['sentistrength_neg_rating']))
                self.data['senti_pos'] = self.data['id'].apply(lambda x: senti_pos_scores[x])
                self.data['senti_neg'] = self.data['id'].apply(lambda x: senti_neg_scores[x])
                print('\tDone!')
            else:
                print('No cached features file found. Computing SentiStrength features...')
                
                print('Setting up SentiStrength module...')
                from sentistrength import PySentiStr
                senti = PySentiStr()
                senti.setSentiStrengthPath(config['sentistrength_jar_path']) 
                senti.setSentiStrengthLanguageFolderPath(config['sentistrength_data_path'])
                print('\tDone.')
                
                # write header
                with open(cached_senti_path,'w') as f:
                    f.write('{}\t{}\t{}\n'.format('post_id','sentistrength_pos_rating',
                                                          'sentistrength_neg_rating'))
                print('Scoring text...')
                # write scores
                with open(cached_senti_path,'a') as f:
                    for _,row in self.data.iterrows():
                        p_id, text = row['id'], row['all_text']
                        pos_score, neg_score = senti.getSentiment(text,score='dual')[0]
                        self.data.at[_,'senti_pos'] = pos_score
                        self.data.at[_,'senti_neg'] = neg_score
                        f.write('{}\t{}\t{}\n'.format(p_id,pos_score,neg_score))
                        if _ % 100 == 0:
                            print('\tOn post {} of {}.'.format(_,self.get_dims()[0]))
                print('\tDone! Cached features have been saved to:',cached_senti_path)
            
        if params['vader']:
            cached_vader_path = os.path.join(cache_prefix,'VADER_scores_concatenated.tsv')
            if os.path.exists(cached_vader_path):
                print('Found cached feature file at {}. Adding cached VADER features...'.format(cached_vader_path))
                vader_df = pd.read_csv(cached_vader_path,sep='\t',header=0)
                vader_df['scores'] = vader_df['scores'].apply(lambda x: json.loads(x))
                vader_scores = dict(zip(vader_df['post_id'],vader_df['scores']))
                self.data['vader_pos'] = self.data['id'].apply(lambda x: vader_scores[x]['pos'])
                self.data['vader_neg'] = self.data['id'].apply(lambda x: vader_scores[x]['neg'])
                self.data['vader_cpd'] = self.data['id'].apply(lambda x: vader_scores[x]['compound'])
                print('\tDone!')
                
            else:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()

                print('No cached features file found. Computing VADER features...')
                with open(cached_vader_path,'w') as f:
                    # write header
                    f.write('{}\t{}\n'.format('post_id','scores'))
                    
                    # write scores
                    for _,row in self.data.iterrows():
                        p_id, text = row['id'], row['all_text']
                        out = analyzer.polarity_scores(text)
                        self.data.at[_,'vader_pos'] = out['pos']
                        self.data.at[_,'vader_neg'] = out['neg']
                        self.data.at[_,'vader_cpd'] = out['compound']
                        f.write('{}\t{}\n'.format(p_id,json.dumps(out)))
                        if _ % 10000 == 0:
                            print('\tOn post {} of {}.'.format(_,self.get_dims()[0]))
                print('\tDone! Cached features have been saved to:',cached_vader_path)
                
        if params['textblob']:
            cached_textblob_path = os.path.join(cache_prefix,'textblob_scores.pkl')
            if os.path.exists(cached_textblob_path):
                print('Found cached feature file at {}. Adding cached TextBlob features...'.format(cached_textblob_path))
                textblob_scores = pickle.load(open(cached_textblob_path,'rb'))
                self.data['tb_sub'] = self.data['id'].apply(lambda x: textblob_scores['subj'][x])
                self.data['tb_pol'] = self.data['id'].apply(lambda x: textblob_scores['pol'][x])
                print('\tDone!')
            else:
                print('No cached features file found. Computing TextBlob features...')
                tb_subjs,tb_pols = {},{}
                for _,row in self.data.iterrows():
                    p_id, text = row['id'], row['all_text']
                    out = TextBlob(text).sentiment
                    subj, pol = out.subjectivity, out.polarity
                    tb_subjs[p_id], tb_pols[p_id] = subj, pol
                    self.data.at[_,'tb_sub'] = subj
                    self.data.at[_,'tb_pol'] = pol
                    if _ % 10000 == 0:
                        print('\tOn post {} of {}.'.format(_,self.get_dims()[0]))
                pickle.dump({'subj':tb_subjs,'pol':tb_pols},open(cached_textblob_path,'wb'))
                print('\tDone! Cached features have been saved to:',cached_textblob_path)
            
        if params['NRC']:
            cached_NRC_path = os.path.join(cache_prefix,'NRC_scores.pkl')
            if os.path.exists(cached_NRC_path):
                print('Found cached feature file at {}. Adding cached NRC features...'.format(cached_NRC_path))
                NRC_scores = pickle.load(open(cached_NRC_path,'rb'))
                self.data['NRC_val'] = self.data['id'].apply(lambda x: NRC_scores[x]['val_sum'])
                self.data['NRC_arousal'] = self.data['id'].apply(lambda x: NRC_scores[x]['arousal_sum'])
                self.data['NRC_dom'] = self.data['id'].apply(lambda x: NRC_scores[x]['dom_sum'])
                print('\tDone!')
            else:
                print('No cached features file found. Computing NRC features...')
                NRC_VAD_df = pd.read_csv(config['NRC_VAD_path'],sep='\t',header=0)
                NRC_dicts = {'valence':dict(zip(NRC_VAD_df['Word'],NRC_VAD_df['Valence'])),
                            'arousal':dict(zip(NRC_VAD_df['Word'],NRC_VAD_df['Arousal'])),
                            'dominance':dict(zip(NRC_VAD_df['Word'],NRC_VAD_df['Dominance']))}
                
                def score_nrc(sent_toks):
                    vals = [NRC_dicts['valence'][t] for t in sent_toks if t in NRC_dicts['valence']]
                    arousals = [NRC_dicts['arousal'][t] for t in sent_toks if t in NRC_dicts['arousal']]
                    doms = [NRC_dicts['dominance'][t] for t in sent_toks if t in NRC_dicts['dominance']]
                    val_sum = sum(vals) #df_['Valence'].sum()/l_
                    arousal_sum = sum(arousals) #df_['Arousal'].sum()/l_
                    dom_sum = sum(doms) #df_['Dominance'].sum()/l_

                    return {'val_sum':val_sum,'arousal_sum':arousal_sum,'dom_sum':dom_sum}

                NRC_scores = {}
                for _,row in self.data.iterrows():
                    p_id, lemmas = row['id'], row['all_lemmas']
                    NRC_scores[p_id] = score_nrc(lemmas)
                    self.data.at[_,'NRC_val'] = NRC_scores[p_id]['val_sum']
                    self.data.at[_,'NRC_arousal'] = NRC_scores[p_id]['arousal_sum']
                    self.data.at[_,'NRC_dom'] = NRC_scores[p_id]['dom_sum']
                    if _ % 10000 == 0:
                        print('\tOn post {} of {}.'.format(_,self.get_dims()[0]))
                pickle.dump(NRC_scores,open(cached_NRC_path,'wb'))
                print('\tDone! Cached features have been saved to:',cached_NRC_path)
            
        if params['wiebe_subjectivity']:
            cached_wiebe_path = os.path.join(cache_prefix,'wiebe_scores.pkl')
            if os.path.exists(cached_wiebe_path):
                print('Found cached feature file at {}. Adding cached Wiebe et al. subjectivity features...'.format(
                    cached_wiebe_path))
                wiebe_scores = pickle.load(open(cached_wiebe_path,'rb'))
                self.data['wiebe_strong'] = self.data['id'].apply(lambda x: wiebe_scores[x]['strongsubj'])
                self.data['wiebe_weak'] = self.data['id'].apply(lambda x: wiebe_scores[x]['weaksubj'])
                print('\tDone!')
            else:
                print('No cached features file found. Computing Wiebe et al. subjectivity features...')
                with open(config['wiebe_path'],'r') as f:
                    subj_clues = f.readlines()
                wiebe_dict = {}
                for line in subj_clues:
                    split_line = line.strip().split(" ")
                    word = split_line[2].split('=')[-1]
                    subjectivity = split_line[0].split('=')[-1]
                    wiebe_dict[word] = subjectivity

                def score_wiebe(sent_toks):
                    subjs_ = [wiebe_dict[t] for t in sent_toks if t in wiebe_dict]
                    return Counter(subjs_)

                wiebe_scores = {}
                for _,row in self.data.iterrows():
                    p_id, lemmas = row['id'], row['all_lemmas']
                    wiebe_scores[p_id] = score_wiebe(lemmas)
                    self.data.at[_,'wiebe_strong'] = wiebe_scores[p_id]['strongsubj']
                    self.data.at[_,'wiebe_weak'] = wiebe_scores[p_id]['weaksubj']
                    if _ % 10000 == 0:
                        print('\tOn post {} of {}.'.format(_,self.get_dims()[0]))
                pickle.dump(wiebe_scores,open(cached_wiebe_path,'wb'))
                print('\tDone! Cached features have been saved to:',cached_wiebe_path)
            
        if params['neg']:
            print('Computing negation feature...')
            def count_neg(x):
                count = 0.0
                for x in re.finditer("not|n't|never|nor|no|nobody|nowhere|nothing|noone",x,re.IGNORECASE):
                    count += 1.0
                return count #/ len(doc.split(" "))
            
            self.data['negation'] = self.data['all_text'].apply(lambda x: count_neg(x))
            print('\tDone!')
            
        if params['nat_disaster']:
            print('Computing natural disaster feature...')
            NATURAL_DISASTER_WORDS = set(open(inputs_path+'/natural_disaster_words.txt','r').read().splitlines())
            def count_nat_disaster(sent_toks):
                return len([t for t in sent_toks if t in NATURAL_DISASTER_WORDS])
        
            self.data['nat_dis'] = self.data['all_lemmas'].apply(lambda x: count_nat_disaster(x))
            print('\tDone!')
            
        if params['economy']:
            print('Computing economy feature...')
            ECONOMY_WORDS = set(open(inputs_path+'/economy_words.txt','r').read().splitlines())
            def count_economy(sent_toks):
                return len([t for t in sent_toks if t in ECONOMY_WORDS])
            
            self.data['econ'] = self.data['all_lemmas'].apply(lambda x: count_economy(x))
            print('\tDone!')
            
        if params['emolex']:
            cached_emo_path = os.path.join(cache_prefix,'emo_intensity_scores.pkl')
            if os.path.exists(cached_emo_path):
                print('Found cached feature file at {}. Adding cached EmoLex features...'.format(cached_emo_path))
                emoint_scores = pickle.load(open(cached_emo_path,'rb'))
                emoint_scores = {p_id: {'intensity':Counter(emoint_scores[p_id]['intensity']),
                                       'counted': emoint_scores[p_id]['counted']}
                                 for p_id in emoint_scores}
                
                for emo in ['anger','anticipation','disgust','fear','joy','sadness','surprise','trust']:
                    self.data['emo_{}'.format(emo)] = self.data['id'].apply(
                        lambda x: emoint_scores[x]['intensity'][emo])
                    self.data['emo_{}_counted'.format(emo)] = self.data['id'].apply(
                        lambda x: emoint_scores[x]['counted'][emo] if emo in emoint_scores[x]['counted'] else [])
                print('\tDone!')
            else:
                print('No cached features file found. Computing EmoLex features...')
                
                emoint = pd.read_csv(config['emoint_path'],sep='\t',header=0)
                emoint_dict = defaultdict(dict)
                for _,row in emoint.iterrows():
                    w, emo, score = row['word'], row['emotion'], row['emotion-intensity-score']
                    emoint_dict[w][emo] = score
                    
                NATURAL_DISASTER_WORDS = set(open(inputs_path+'/natural_disaster_words.txt','r').read().splitlines())
                ECONOMY_WORDS = set(open(inputs_path+'/economy_words.txt','r').read().splitlines())
                    
                def score_emo_intensity(sent_toks):
                    assocs_, overlap_  = Counter(), defaultdict(list)
                    for ix_t,t in enumerate(sent_toks):
                        if t in emoint_dict and t not in BLACKLIST_WORDS and \
                        t not in NATURAL_DISASTER_WORDS and t not in ECONOMY_WORDS:
                            if t == 'green' and ix_t < len(sent_toks)-2 and \
                            sent_toks[ix_t+1] == 'new' and sent_toks[ix_t+2] == 'deal':
                                pass
                            elif t == 'new' and ix_t > 0 and ix_t < len(sent_toks)-1 and \
                            sent_toks[ix_t-1] == 'green' and sent_toks[ix_t+1] == 'deal':
                                pass
                            elif t == 'deal' and ix_t > 1 and \
                            sent_toks[ix_t-1] == 'new' and sent_toks[ix_t-2] == 'green':
                                pass
                            elif (t == 'united' or t == 'united') and ix_t < len(sent_toks)-1 and \
                            (sent_toks[ix_t+1] == 'nation' or sent_toks[ix_t+1] == 'nations'):
                                pass
                            elif (t == 'nation' or t == 'nations') and ix_t > 0 and \
                            (sent_toks[ix_t-1] == 'unite' or sent_toks[ix_t-1] == 'united'):
                                pass
                            else:
                                assocs_ += Counter(emoint_dict[t]) 
                                for emo in emoint_dict[t]:
                                    overlap_[emo].append(t)
                    return {'intensity': assocs_,'counted': overlap_}
                
                emoint_scores = {}
                for _,row in self.data.iterrows():  
                    p_id, lemmas = row['id'], row['all_lemmas']
                    emoint_scores[p_id] = score_emo_intensity(lemmas)
                    if _ % 10000 == 0:
                        print('\tOn post {} of {}.'.format(_,self.get_dims()[0]))
                pickle.dump(emoint_scores,open(cached_emo_path,'wb'))
                
                for emo in ['anger','anticipation','disgust','fear','joy','sadness','surprise','trust']:
                    self.data['emo_{}'.format(emo)] = self.data['id'].apply(
                        lambda x: emoint_scores[x]['intensity'][emo])
                    self.data['emo_{}_counted'.format(emo)] = self.data['id'].apply(
                        lambda x: emoint_scores[x]['counted'][emo] if emo in emoint_scores[x]['counted'] else [])
                print('\tDone! Cached features have been saved to:',cached_emo_path)
            
        if params['morals']:
            cached_mfd_path = os.path.join(cache_prefix,'MFD_scores_per_post.tsv')
            if os.path.exists(cached_mfd_path):
                print('Found cached feature file at {}. Adding cached Moral Foundations features...'.format(cached_mfd_path))
                morals_df = pd.read_csv(cached_mfd_path,sep='\t',header=0)
                morals_dict = dict(zip(morals_df['post_id'],morals_df['moral_foundation_counts'].apply(
                    lambda x: json.loads(x))))
                morals_dict = {p_id: {'score': Counter(morals_dict[p_id]['score']),
                                     'counted': morals_dict[p_id]['counted']} for p_id in morals_dict}
                for mf in ['purity','harm','authority','fairness','loyalty']:
                    for mf_val in ['vice','virtue']:
                        key_ = '{}_{}'.format(mf_val,mf)
                        self.data[key_] = self.data['id'].apply(lambda x: morals_dict[x]['score'][key_])
                        self.data['{}_counted'.format(key_)] = self.data['id'].apply(
                            lambda x: morals_dict[x]['counted'][key_] if key_ in morals_dict[x]['counted'] else [])
                print('\tDone!')
            else:
                print('No cached features file found. Computing Moral Foundations features...')
                
                import xlrd
                MFD_df = pd.read_excel(config['moral_foundations_path'])
                MFD_df['valence+foundation'] = MFD_df['valence']+'_'+MFD_df['foundation']
                word2foundation = dict(zip(MFD_df['word'],MFD_df['valence+foundation']))
                NATURAL_DISASTER_WORDS = set(open(inputs_path+'/natural_disaster_words.txt','r').read().splitlines())
                ECONOMY_WORDS = set(open(inputs_path+'/economy_words.txt','r').read().splitlines())
                
                def score_MFD(sent_toks):
                    founds_, overlap_ = Counter(), defaultdict(list)
                    for ix_t,t in enumerate(sent_toks):
                        if t in word2foundation and t not in BLACKLIST_WORDS and \
                        t not in NATURAL_DISASTER_WORDS and t not in ECONOMY_WORDS:
                            if t == 'green' and ix_t < len(sent_toks)-2 and \
                            sent_toks[ix_t+1] == 'new' and sent_toks[ix_t+2] == 'deal':
                                pass
                            elif t == 'new' and ix_t > 0 and ix_t < len(sent_toks)-1 and \
                            sent_toks[ix_t-1] == 'green' and sent_toks[ix_t+1] == 'deal':
                                pass
                            elif t == 'deal' and ix_t > 1 and \
                            sent_toks[ix_t-1] == 'new' and sent_toks[ix_t-2] == 'green':
                                pass
                            elif (t == 'united' or t == 'united') and ix_t < len(sent_toks)-1 and \
                            (sent_toks[ix_t+1] == 'nation' or sent_toks[ix_t+1] == 'nations'):
                                pass
                            elif (t == 'nation' or t == 'nations') and ix_t > 0 and \
                            (sent_toks[ix_t-1] == 'unite' or sent_toks[ix_t-1] == 'united'):
                                pass
                            else:
                                founds_[word2foundation[t]] += 1
                                overlap_[word2foundation[t]].append(t)
                    return {'score': founds_,'counted': overlap_}
                
                with open(cached_mfd_path,'w') as f:
                    # write header
                    f.write('{}\t{}\n'.format('post_id','moral_foundation_counts'))
                    # write scores
                    for _,row in self.data.iterrows():  
                        p_id, lemmas = row['id'], row['all_lemmas']
                        f.write("{}\t{}\n".format(p_id,json.dumps(score_MFD(lemmas))))
                        if _ % 10000 == 0:
                            print('\tOn post {} of {}.'.format(_,self.get_dims()[0]))
                            
                morals_df = pd.read_csv(cached_mfd_path,sep='\t',header=0)
                morals_dict = dict(zip(morals_df['post_id'],morals_df['moral_foundation_counts'].apply(
                    lambda x: json.loads(x))))
                morals_dict = {p_id: {'score': Counter(morals_dict[p_id]['score']),
                                     'counted': morals_dict[p_id]['counted']} for p_id in morals_dict}
                for mf in ['purity','harm','authority','fairness','loyalty']:
                    for mf_val in ['vice','virtue']:
                        key_ = '{}_{}'.format(mf_val,mf)
                        self.data[key_] = self.data['id'].apply(lambda x: morals_dict[x]['score'][key_])
                        self.data['{}_counted'.format(key_)] = self.data['id'].apply(
                            lambda x: morals_dict[x]['counted'][key_] if key_ in morals_dict[x]['counted'] else [])
                print('\tDone! Cached features have been saved to:',cached_mfd_path)
                
        if params['values']:
            cached_values_path = os.path.join(cache_prefix,'human_value_scores_per_post.tsv')
            if os.path.exists(cached_values_path):
                print('Found cached feature file at {}. Adding cached Basic Human Values features...'.format(
                    cached_values_path))
                hum_values_df = pd.read_csv(cached_values_path,sep='\t',header=0)
                hum_values_dict = dict(zip(hum_values_df['post_id'],hum_values_df['human_value_counts'].apply(
                    lambda x: json.loads(x))))
                hum_values_dict = {p_id: {'score': Counter(hum_values_dict[p_id]['score']),
                                     'counted': hum_values_dict[p_id]['counted']} for p_id in hum_values_dict}
                for value in ['self_direction','stimulation','hedonism','achievement','power',
                             'security','conformity','tradition','benevolence','universalism']:
                    self.data[value] = self.data['id'].apply(lambda x: hum_values_dict[x]['score'][value])
                    self.data['{}_counted'.format(value)] = self.data['id'].apply(
                        lambda x: hum_values_dict[x]['counted'][value] 
                        if value in hum_values_dict[x]['counted'] else [])
                print('\tDone!')
            else:
                print('No cached features file found. Computing Basic Human Values features...')
                values_dict = pickle.load(open(inputs_path+'/  ','rb'))
                word2value = {w: value for value in values_dict for w in values_dict[value]}
                NATURAL_DISASTER_WORDS = set(open(inputs_path+'/natural_disaster_words.txt','r').read().splitlines())
                ECONOMY_WORDS = set(open(inputs_path+'/economy_words.txt','r').read().splitlines())
                
                def score_basic_hum_value(sent_toks):
                    values_, overlap_ = Counter(), defaultdict(list)
                    for ix_t,t in enumerate(sent_toks):
                        if t in word2value and t not in BLACKLIST_WORDS and \
                        t not in NATURAL_DISASTER_WORDS and t not in ECONOMY_WORDS:
                            if t == 'green' and ix_t < len(sent_toks)-2 and \
                            sent_toks[ix_t+1] == 'new' and sent_toks[ix_t+2] == 'deal':
                                pass
                            elif t == 'new' and ix_t > 0 and ix_t < len(sent_toks)-1 and \
                            sent_toks[ix_t-1] == 'green' and sent_toks[ix_t+1] == 'deal':
                                pass
                            elif t == 'deal' and ix_t > 1 and \
                            sent_toks[ix_t-1] == 'new' and sent_toks[ix_t-2] == 'green':
                                pass
                            elif (t == 'united' or t == 'united') and ix_t < len(sent_toks)-1 and \
                            (sent_toks[ix_t+1] == 'nation' or sent_toks[ix_t+1] == 'nations'):
                                pass
                            elif (t == 'nation' or t == 'nations') and ix_t > 0 and \
                            (sent_toks[ix_t-1] == 'unite' or sent_toks[ix_t-1] == 'united'):
                                pass
                            else:
                                values_[word2value[t]] += 1
                                overlap_[word2value[t]].append(t)
                    return {'score': values_, 'counted': overlap_}
                
                with open(cached_values_path,'w') as f:
                    # write header
                    f.write('{}\t{}\n'.format('post_id','human_value_counts'))
                    # write scores
                    for _,row in self.data.iterrows():  
                        p_id, lemmas = row['id'], row['all_lemmas']
                        f.write("{}\t{}\n".format(p_id,json.dumps(score_basic_hum_value(lemmas))))
                        if _ % 10000 == 0:
                            print('\tOn post {} of {}.'.format(_,self.get_dims()[0]))
                            
                hum_values_df = pd.read_csv(cached_values_path,sep='\t',header=0)
                hum_values_dict = dict(zip(hum_values_df['post_id'],hum_values_df['human_value_counts'].apply(
                    lambda x: json.loads(x))))
                hum_values_dict = {p_id: {'score': Counter(hum_values_dict[p_id]['score']),
                                     'counted': hum_values_dict[p_id]['counted']} for p_id in hum_values_dict}
                for value in ['self_direction','stimulation','hedonism','achievement','power',
                             'security','conformity','tradition','benevolence','universalism']:
                    self.data[value] = self.data['id'].apply(lambda x: hum_values_dict[x]['score'][value])
                    self.data['{}_counted'.format(value)] = self.data['id'].apply(
                        lambda x: hum_values_dict[x]['counted'][value] 
                        if value in hum_values_dict[x]['counted'] else [])
                print('\tDone! Cached features have been saved to:',cached_values_path)
    
    def get_categorical_features(self, threshold='median', **params):
        """
        Creates categorical versions of continuous features as new columns in `self.data`.
        """
        print('\nGetting categorical versions of features...')
        print('Creating thresholds for each feature based on {} of non-zero values...'.format(threshold))
        if threshold == 'median':
            non_zero_thresholds = {feat: self.data.loc[self.data[feat]>0][feat].median()
                   for feat in params}
        if threshold == 'mean':
            non_zero_thresholds = {feat: self.data.loc[self.data[feat]>0][feat].mean()
                   for feat in params}
            
        print('Creating categorical features...')
        for feat in params:
            if feat == 'senti_pos':
                self.data['senti_pos_cat'].apply(lambda x: 1 if x > 1 else 0)
            if feat == 'senti_neg':
                self.data['senti_neg_cat'].apply(lambda x: 1 if x > 1 else 0)
            else:
                self.data[feat+'_cat'] = self.data[feat].apply(lambda x: 1 if x >= non_zero_thresholds[feat] else 0)
        print('\tDone!')
    
    def set_features(self, ipython_disp=False):
        """
        Sets `self.feats_dict` to key (feature category), val (feature name) pairs to help keep track of all the different 
        features.
        
        """
        
        # All possible features, categorized by feature type
        emoint_emotions = ['ant','joy','sadness','disgust','anger','surprise','fear','trust']
        mfd_foundations = ['purity','harm','authority','fairness','loyalty']
        hum_values = ['self_direction','stimulation','hedonism','achievement','power',
                      'security','conformity','tradition','benevolence','universalism']
        
        length_feats = ['log_len'] 
        affect_feats = ['senti_pos','senti_neg',
                        'vader_pos','vader_neg','vader_cpd',
                        'tb_pol',
                        'NRC_val','NRC_arousal','NRC_dom']+\
                       ['emo_{}'.format(emo) for emo in emoint_emotions]+\
                       ['vice_{}'.format(mf) for mf in mfd_foundations]+\
                       ['virtue_{}'.format(mf) for mf in mfd_foundations]+\
                        hum_values
        affect_cat_feats = ['{}_cat'.format(feat) for feat in affect_feats 
                           if feat != 'vader_cpd' and feat != 'tb_pol']
        affect_cat_feats.extend(['vader_cpd','tb_pol'])
        ling_feats = ['tb_sub','wiebe_strong','wiebe_weak','negation','nat_dis','econ']
        ling_cat_feats = ['{}_cat'.format(feat) for feat in ling_feats]
        url_feats = [feat for feat in self.data.columns if feat.startswith('is_') 
                     and feat != 'is_video' and feat != 'is_link_post']
        url_type_feats = ['is_right','is_right_center','is_center','is_left_center','is_left',
                          'is_sci','is_conspiracy','is_questionable']
        url_domain_feats = list(set(url_feats).difference(set(url_type_feats)))
        people_feats = [feat for feat in self.data.columns if feat.startswith('has_')]
        time_feats = ['year']
        
        feats_no_cats = length_feats + affect_feats + ling_feats + url_feats + people_feats + time_feats
        feats_with_cats = length_feats + affect_cat_feats + ling_cat_feats + url_feats + people_feats + time_feats

        self.feats_dict = {
            'length': length_feats,
            'affect': affect_feats,
            'affect_cat': affect_cat_feats,
            'ling': ling_feats,
            'ling_cat': ling_cat_feats,
            'all_url': url_feats,
            'url_type': url_type_feats,
            'url_domain': url_domain_feats,
            'people': people_feats,
            'time': time_feats,
            'categorical': feats_with_cats,
            'non_categorical': feats_no_cats
        }
        
        # Subset to those that have been computed for the data
        for feat_type in self.feats_dict:
            tl = len(self.feats_dict[feat_type])
            self.feats_dict[feat_type] = set(self.feats_dict[feat_type]).intersection(set(self.data.columns))
            print('Found {} out of {} possible {} features.'.format(len(self.feats_dict[feat_type]),
                                                                      tl,feat_type.upper()))
        
        all_feats = list(set(self.feats_dict['categorical']) | set(self.feats_dict['non_categorical']))
        print('\nFound {} total features in data.'.format(len(all_feats)))
        if ipython_disp:
            display(pd.DataFrame.from_dict(self.feats_dict,orient='index').T)
        
    def get_zscores(self):
        """Computes z-scores for all features."""
        
        print('\nGetting zscores...')
        zscored_feats = []
        for feat_type in self.feats_dict:
            for feat in self.feats_dict[feat_type]:
                if feat == 'year':
                    year_mean = self.data['year'].mean()
                    year_std = self.data['year'].std()
                    self.data['{}_zscore'.format(feat)] = self.data['year'].apply(lambda x: (x-year_mean)/year_std)
                else:
                    self.data['{}_zscore'.format(feat)] = stats.zscore(self.data[feat])
                zscored_feats.append('{}_zscore'.format(feat))
        print('\tDone!')
        
        # Update self.feats_dict
        self.feats_dict['zscored'] = zscored_feats
    
    def get_residuals(self,resid_by='log_len'):
        """Computes residuals for all features (except log(length)) using log(length)."""
        
        print('\nGetting residuals...')
        resid_feats = []
        for col in self.data.columns:
            if col.endswith('_zscore') and col != 'log_len_zscore':
                self.data[col+'_resid'] = sm.ols(formula='{} ~ {}'.format(col,resid_by), 
                                             data=self.data).fit().resid
                resid_feats.append(col+'_resid')
        print('\tDone!')
        
        self.feats_dict['resid'] = resid_feats
        
    def plot_features(self,ipython_disp=False):
        """Creates plots showing feature distributions (histograms for continuous, piecharts for categorical)"""
        
        print('\nCreating figures...')
        fig_dir = os.path.join(config['base_dir'],self.out_dir,'figs')
        if not(os.path.exists(fig_dir)):
            print('\tMaking new directory for figures:',fig_dir)
            os.mkdir(fig_dir)
        
        for feat_type in ['length','affect','affect_cat','ling','url','people','time']:
            for feat in self.feats_dict[feat_type]:
                fig = sns.distplot(self.data[feat])
                fig.figure.savefig(os.path.join(fig_dir,'{}_histogram.png'.format(feat)))
                plt.clf()
                if ipython_disp:
                    img = Image(os.path.join(fig_dir,'{}_histogram.png'.format(feat)))
                    display(img)
    
        post_ids_per_cat = {cat: self.data.loc[self.data[cat]==1]['id'] for cat in self.feats_dict['url_type_feats']}
        post_id2dom_cats = defaultdict(list)
        for cat in post_ids_per_cat:
            for p_id in post_ids_per_cat[cat]:
                post_id2dom_cats[p_id].append(cat)
        self.data['media_cats'] = self.data['id'].apply(
            lambda x: post_id2dom_cats[x] if x in post_id2dom_cats else None)
        counts_ = Counter([item for sublist in self.data['media_cats'].dropna() for item in sublist])
        plt.pie([float(v) for v in counts_.values()], labels=[k for k in counts_],
           autopct=None)
        plt.savefig(os.path.join(fig_dir,'URL_categories_pieplot.png'))
        plt.clf() 
        if ipython_disp:
            display(Image(filename=os.path.join(fig_dir,'URL_categories_pieplot.png')))
            
        most_common_people = set(open(os.path.join(config['base_dir'],config['data_dir'],
                                                   'most_common_people.txt'),'r').read().splitlines())
        post_ids_per_person = {cat: self.data.loc[self.data[cat]==1]['id'] for cat in 
                   ['has_{}'.format(p.replace('-','')) for p in most_common_people]}
        post_id2people = defaultdict(list)
        for cat in post_ids_per_person:
            for p_id in post_ids_per_person[cat]:
                post_id2people[p_id].append(cat)
        people_mentioned = self.data['id'].apply(lambda x: post_id2people[x] if x in post_id2people else None)
        people_counts_ = Counter([item for sublist in people_mentioned.dropna() for item in sublist])
        plt.pie([float(v) for v in people_counts_.values()], labels=[k for k in people_counts_],
           autopct=None)
        plt.savefig(os.path.join(fig_dir,'people_mentioned_pieplot.png'))
        plt.clf() 
        if ipython_disp:
            display(Image(filename=os.path.join(fig_dir,'people_mentioned_pieplot.png')))
        
class LinReg:

    def __init__(self, base_df, base_df_name, ind_vars, dep_vars=['log_num_comments','log_score','ratio_comments'], 
                 subsets=['all_posts','posts_without_links','conservative_posts','non_conservative_posts',
                          'non_zero_engagement_posts']):
        self.base_df = base_df
        self.base_df_name = base_df_name
        self.ind_vars = ind_vars
        self.collinear_feats = None
        self.dep_vars = dep_vars
        self.formulas = None
        self.subsets = subsets
        self.data_dir = config['data_dir']
        print('\nCreating a regression for {} with {} features.'.format(self.base_df_name.upper(),
              len(self.ind_vars)))
        if len(self.subsets) > 0:
            print('Additional regressions for the following subsets of posts will be created:',subsets)
        self.sig_results = None
        
        
    def get_base_df(self):
        return self.base_df
    
    def get_base_df_name(self):
        return self.base_df_name
    
    def get_ind_vars(self):
        return self.ind_vars
    
    def get_collinear_feats(self):
        return self.collinear_feats
    
    def get_dep_vars(self):
        return self.dep_vars
    
    def get_formulas(self):
        return self.formulas
    
    def get_subsets(self):
        return self.subsets
    
    def get_significant_results(self):
        """Returns summary of significant results."""
        return self.sig_results
    
    def get_VIF(self, threshold=5):
        """Calculates the variance inflation factor for each feature to determine which are collinear."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from patsy import dmatrices
        
        print('\nCalculating VIF scores (this takes a while)...')
        features = "+".join(self.ind_vars)
        # choice of Y does not matter
        y_, X_ = dmatrices('log_num_comments ~' + features, self.base_df, return_type='dataframe')
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])]
        vif["feature"] = X_.columns
        print(tabulate(vif.round(2).sort_values('VIF Factor')))
        self.collinear_feats = vif.loc[vif["VIF Factor"]>=threshold]["feature"].values
        print('\tFound {} collinear features.'.format(len(self.collinear_feats)))
        
    def get_Ys(self):
        """Loads and prepares the dependent variables for regression."""
        print('\nGetting the log transforms of the following dependent variables:',self.dep_vars)
        for y in self.dep_vars:
            if y == 'log_num_comments':
                self.base_df['log_num_comments'] = self.base_df['num_comments'].apply(lambda x: math.log(x+0.001))
            elif y == 'log_score':
                self.base_df['log_score'] = self.base_df['score'].apply(lambda x: math.log(x+0.001))
            elif y == 'ratio_comments':
                comment_sent_ratios_df = pd.read_csv(os.path.join(
                    self.data_dir,'post_comment_sent_ratios.tsv'),sep='\t',header=0)
                p_id2ratio_comments = dict(zip(comment_sent_ratios_df['post_id'],comment_sent_ratios_df['ratio_pos_neg']))
                self.base_df['ratio_comments'] = self.base_df['id'].apply(lambda x: p_id2ratio_comments[x])
            else:
                pass
        print('\tDone!')
        
    def get_formulas(self):
        """Creates regression formulas, excluding collinear variables."""
        print('\nGetting regression formulas...')
        formulas = []
        for y in self.dep_vars:
            if len(self.collinear_feats) == 0:
                formulas.append('{} ~ {}'.format(y,' + '.join(self.ind_vars)))
            else:
                for feat in self.collinear_feats:
                    # keep only 1 feature among collinear features
                    formulas.append('{} ~ {}'.format(
                        y,' + '.join([x for x in self.ind_vars if x not in self.collinear_feats or x == feat])))
        self.formulas = formulas
        #print(formulas)
        
    def get_sub_frames(self):
        """Creates subframes from `self.base_df` according to criteria specified in `self.subsets`."""
        
        print('\nGetting subframes for regression...')
        conservative_subs = set(open(self.data_dir+'/conservative_subs.txt','r').read().splitlines())
        conservative_subs = set([x.lower() for x in conservative_subs])
        religion_subs = set(open(self.data_dir+'/religion_subs.txt','r').read().splitlines())
        religion_subs = set([x.lower() for x in religion_subs])
        
        for subset in self.subsets:
            if subset == 'posts_without_links':
                self.no_links_df = self.base_df.loc[self.base_df['is_link_post']==False].copy()
                print('Created subframe of {}, with dimensions:'.format(subset.replace('_',' ')),
                                                                       self.no_links_df.shape)
            if subset == 'conservative_posts':
                self.conservative_df = self.base_df.loc[
                    self.base_df['subreddit'].apply(lambda x: x.lower() if x else '').isin(conservative_subs)].copy()
                print('Created subframe of {}, with dimensions:'.format(subset.replace('_',' ')),
                                                                       self.conservative_df.shape)
            if subset == 'non_conservative_posts':
                self.non_conservative_df = self.base_df.loc[~self.base_df['subreddit'].apply(
                    lambda x: x.lower() if x else '').isin(conservative_subs | religion_subs)].copy()
                print('Created subframe of {}, with dimensions:'.format(subset.replace('_',' ')),
                                                                       self.non_conservative_df.shape)
            if subset == 'non_zero_engagement_posts':
                self.nonzero_eng_df = pd.concat([self.base_df.loc[self.base_df['num_comments'] > 0].copy(),
                                          self.base_df.loc[self.base_df['ratio_comments'] > 0].copy(),
                                          self.base_df.loc[self.base_df['score'] > 0].copy()])
                self.nonzero_eng_df = self.nonzero_eng_df.drop_duplicates('id',keep='first')
                print('Created subframe of {}, with dimensions:'.format(subset.replace('_',' ')),
                                                                       self.non_conservative_df.shape)
    
    def fit(self,formula_,reg_df_):
        """Calls the statsmodels `fit` function and returns resulting model."""
        fit_ = sm.ols(formula=formula_, data=reg_df_).fit()
        return fit_

    def batch_regress(self,out_dir): 
        """Regresses different subsets of data, and also loops over different formulas."""
        print('\nDoing regressions...')
        results_list = []
        for _,subset in enumerate(self.subsets+['all_posts']):
            print('Regressing subset with:',subset.replace('_',' ').upper())
            if subset == 'all_posts':
                reg_df = self.base_df
            elif subset == 'posts_without_links':
                reg_df = self.no_links_df
            elif subset == 'conservative_posts':
                reg_df = self.conservative_df
            elif subset == 'non_conservative_posts':
                reg_df = self.non_conservative_df
            elif subset == 'non_zero_engagement_posts':
                reg_df = self.nonzero_eng_df
                
            for formula in self.formulas:
                result = self.fit(formula, reg_df)
                df_ = pd.concat([result.params,result.pvalues],axis=1)
                df_.columns = ['coeff','pvalue']
                df_['subset'] = [subset]*len(df_)
                df_['y_var'] = [formula.split(' ~ ')[0]]*len(df_)
                df_['formula'] = [formula]*len(df_)
                results_list.append(df_)
        
        results_df = pd.concat(results_list,axis=0)
        sig_results_df = results_df.loc[(results_df['pvalue']<0.001)].copy()
        print('Found {} significant coefficients from {} coefficients total.'.format(
            len(sig_results_df),len(results_df)))
        sig_results_df['predictor'] = sig_results_df.index.copy()
        sig_results_df.reset_index(drop=True,inplace=True)
        sig_results_df = sig_results_df.loc[sig_results_df['predictor']!='Intercept']
        sig_results_df['pretty_pred'] = sig_results_df['predictor'].apply(
            lambda x: x.replace('_zscore','').replace('_resid',''))
        self.sig_results = sig_results_df
        self.sig_results.to_csv(os.path.join(out_dir,'sig_results.tsv'),
                                sep='\t',header=True,index=False)
    
    def plot_coefficients(self,feature_set,subset,savename):
        """Plots the coefficients of regression."""
        
        data_to_plot = self.sig_results.loc[self.sig_results['pretty_pred'].isin(feature_set)]
        data_to_plot = data_to_plot.loc[data_to_plot['subset']==subset]
        my_order = data_to_plot.groupby(by=['pretty_pred'])['coeff'].median().sort_values(ascending=False).index

        fig,ax = plt.subplots(figsize=(20,8))
        sns.boxplot(x='pretty_pred',y='coeff',
                    data=data_to_plot,
                    ax=ax,color='blanchedalmond',order=my_order)
        sns.stripplot(x='pretty_pred',y='coeff',hue='y_var',
                    data=data_to_plot,
                    ax=ax,jitter=0.5,s=8,linewidth=0.001,alpha=1,order=my_order)
        ax.set_ylabel("Coefficient",fontsize=24)
        ax.set_xlabel("",fontsize=24)
        for ax in fig.axes:
            ax.tick_params(labelrotation=90,labelsize=20)
        plt.title('Multi regression results on {}'.format(subset),fontsize=28)
        plt.legend(fontsize='x-large', title_fontsize='40')
        plt.tight_layout()
        fig.savefig(savename+'.png')
        
        
if __name__ == '__main__':
    
    DG = DataGetter(base_dir='/juice/scr/yiweil/green-lexicon',data_dir='reddit_data',data_fname='posts_with_words.pkl')
    DG.load_data()
    DG.filter_data()
    DG.get_features(length=True, url=True, top_N_domains=20, people=True, sentistrength=False, vader=False, 
                    textblob=True, wiebe_subjectivity=False, NRC=False, neg=True, nat_disaster=True,
                    economy=True, emolex=True, morals=True, values=True)
    DG.get_categorical_features(tb_sub=True,
                               emo_anger=True,emo_anticipation=True,emo_disgust=True,emo_fear=True,
                                emo_joy=True,emo_sadness=True,emo_surprise=True,emo_trust=True,
                                negation=True,nat_dis=True,econ=True,
                               vice_purity=True,vice_harm=True,vice_authority=True,vice_fairness=True,vice_loyalty=True,
                              virtue_purity=True,virtue_harm=True,virtue_authority=True,virtue_fairness=True,virtue_loyalty=True,
                                self_direction=True,stimulation=True,hedonism=True,achievement=True,power=True,
                               security=True,conformity=True,tradition=True,benevolence=True,universalism=True)
    DG.set_features()
    DG.get_zscores()
    DG.get_residuals()
    DG.plot_features(ipython_disp=True)
    
    dg_df = DG.get_data()
    dg_df.shape
    
    all_feats = [x for x in dg_df.columns if 'zscore_resid' in x or x == 'log_len_zscore']
    all_cat_feats = [x for x in all_feats if '_cat' in x]
    cat_feats_only = []
    visited = {}

    for feat in all_cat_feats:
        non_cat_version = feat.replace('_cat','')
        assert non_cat_version in all_feats
        visited[non_cat_version] = 0
        visited[feat] = 0
        cat_feats_only.append(feat)

    for feat in all_feats:
        if feat not in visited:
            cat_feats_only.append(feat)
            
    LR = LinReg(base_df=dg_df,base_df_name='all_posts',
                ind_vars=cat_feats_only,dep_vars=['log_num_comments','log_score','ratio_comments'],
           subsets=['posts_without_links','conservative_posts','non_conservative_posts','non_zero_engagement_posts'],
           data_dir='/juice/scr/yiweil/green-lexicon/reddit_data')
    LR.get_VIF()
    LR.get_Ys()
    LR.get_formulas()
    LR.get_sub_frames()
    LR.batch_regress()
    LR.get_significant_results().to_csv(os.path.join(DG.__dict__['out_dir'],'sig_results.tsv'),sep='\t',header=True,index=False)
    
    # do some plotting
    for subset in LR.__dict__['subsets']:
        for feat_type in ['length', 'affect_cat', 'ling_cat', 'url', 'people', 'time']:
            feature_set = DG.__dict__['feats_dict'][feat_type]
            savename = os.path.join(DG.__dict__['out_dir'],'figs','{}_{}_effects'.format(subset,feat_type))
            LR.plot_coefficients(feature_set,subset,savename)
    