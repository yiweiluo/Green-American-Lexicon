# Analysis of Reddit posts relating to climate change

## Getting started:
1. Run `pip install -r requirements.txt` to get all packages.
2. Update `config.json` so that the path for `base_dir` is the location of this repository. For the other paths, you can leave them as is if you are using this repository on NLP cluster, otherwise, you should `scp` the data from those locations to somewhere you'll be able to read them from, and update their paths accordingly. 
3. Download the file `posts_with_words.pkl.zip` (containing all of the Reddit climate change data) from [here](https://drive.google.com/file/d/1z29MgH2WGN0JN8R5r07r6MUV1CCpzKF0/view?usp=sharing) and unzip it to the `reddit_data` subdirectory.

* **A. Identifying words correlated with high engagement**
	1. Get data w/ signals of engagement using appropriate APIs for:
		- [x] tweets about climate change etc. (`/u/scr/nlp/twitter` dump)
		- [x] posts from subreddits about climate change etc.
			- [x] Reddit bot detection
		- other potential sources:
			- [ ] broadcast TV news dataset (also their visualizer: https://tvnews.stanford.edu/)
			- [ ] White House press releases
			- [Chevron-sponsored energy research](https://cisoft.usc.edu/uscchevron-frontiers-of-energy-resources-summer-camp/)
	2. Explore data
		- [x] Understand subreddits qualitatively
		- [x] Divide up Reddit data into pro- and anti- climate change
		- [ ] Log odds analyses (overall, by engagement, by stance)
			- [ ] fix stopword/pronoun bug
			* try length as engagement signal first (then others)
				- [ ] take into account comment length in engagement metric
				- [ ] take into account follower count in engagement metric
			* different partioning methods for low vs. high engagement posts
				* thresholding function that’s dependent on a given sub’s engagement levels (mean, std.)
				* absolute threshold (e.g. log(#comments) > 2; some subs may just write more engaging posts than others) -- Log Odds should be robust to choice of threshold 
				* exclude middle (only look at extremes)
			* apply other potential categories/filters to data comparisons 
				- [ ] hope vs. concern (/fear)--NRC and SentiStrength; who's using and who's responding
				* e.g. whether original post mentions climate change
				* restrict word type (names, adjectives)
				* restrict to words occurring in multiple subreddits
	3. Apply Reid's deconfounded lexicon induction model
		- [ ] decide on potential confounding variables
* **B. Identifying effective words**
	1. Review literature on effective climate change communication to compile [master list of frames/strategies](https://docs.google.com/spreadsheets/d/1GEhVp_Yo9GPCnbvWYxIqJJRE556adaUOVjeBqd5Lky0/edit#gid=0)
		- [ ] Framing literature (economic cost, (scientific) uncertainty)
			- [ ] Gabrielle Wong-Parodi
			- [ ] [add more]
		- [ ] Interviews Zach conducted with climate change NGOs and practitioners
	4. Operationalize effective strategies computationally
		- [ ] Create/curate lexicons associated with given frames/strategies 
	5. Train word embedding models
		- [ ] Gather datasets (news articles, tweets, Reddit posts)
		- [ ] Stratify datasets according to variables like attitude, audience, author
	6. Query models for replacement terms
* **C. Packaging components A+B in easy-to-use interface**

## Potential project timeline:
1. Data collection and pre-processing for components A. and B. (*2 weeks; July 13-July 27*)
2. Data exploration (*1 week; July 27-August 3*)
3. Component A. (run Reid's code) (*1/2 week; August 3-August 5*)
4. Component B. 
	* i. Lit review (*ongoing weeks 1-3; July 13-August 3*)
	* ii. Create some seed lexicons (*1 week; August 6-August 12*)
	* iii.-iv. Train and query VSMs (*3 weeks; August 13-September 2*)
5. Create model interfaces (*rest of September*)

## Old:
- [x] Reach out to Allison Koenecke about using off-the-shelf tweet classifier
- [x] Classify Harold et al's tweet data
- [x] Retrain language models for climate change affirmers, non-affirmers
- [ ] Deconfounded lexicon induction for engagement prediction

## Ideas for TODOs:

- [ ] Context-aware engagement prediction
  - [ ] Deconfounded lexicon induction (https://nlp.stanford.edu/pubs/pryzant2018lexicon.pdf)
  - [ ] Weight engagement function by some kind of similarity function between target and original context(s)
- [ ] Stance awareness (target = "Climate change is a real concern")
  - [ ] Intuition: want to maximize engagement score for tweets with pro-target stance and minimize engagement score for tweets with anti-target stance
  - [ ] Build classifier to classify stance of a tweet (based on: text features, account following network)
- [ ] Other algorithm tweaks
  - [ ] Thesaurus sources beyond WordNet to generate alternative candidates
  - [ ] Making use of language models: learn embedding space for each side (anti-/pro- climate change being a real concern) and propose nearest neighbors of positive sentiment words ("inspiration", "awesome", "economic growth") as candidates
  - [ ] Or weight engagement function by cosine similarity between candidate word and set of positive sentiment words (or some other LIWC category--social processes, achievement, etc.) 
- [ ] Create evaluation metrics
  - [ ] Test data: manually annotate a set of tweet-length sentence pairs (differing in a candidate word/phrase) for their relative improvement
  - [ ] Look into other sources

## References:
- Deconfounded Lexicon Induction: https://nlp.stanford.edu/pubs/pryzant2018lexicon.pdf
- SemEval stance prediction task: http://alt.qcri.org/semeval2016/task6/index.php?id=data-and-tool
- LIWC: https://www.kovcomp.co.uk/wordstat/LIWC.html
	- http://lit.eecs.umich.edu/~geoliwc/LIWC_Dictionary.htm
	- Python wrappers: https://github.com/chbrown/liwc-python, https://pypi.org/project/liwc-analysis/
- [The New American Lexicon (book page 165; this is the document advising the Republican party on language change)](https://joshuakahnrussell.files.wordpress.com/2008/10/luntzplaybook2006.pdf)
  - A few basic papers on sentiment analysis for climate change language:
    - [Climate Change Sentiment on Twitter: An Unsolicited Public Opinion Poll (highest citation count paper on the subject that I could find)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4546368/)
    - [Tracking Climate Change Opinions from Twitter Data](https://pdfs.semanticscholar.org/0a20/18c2a701d72d0ded2a9f58faf49f34099e81.pdf)
    - [Climate Sentiment Analysis on News Data (R notebook) (perhaps not the most useful analysis, esp because in R, but perhaps interesting to look at their datasets)](https://rstudio-pubs-static.s3.amazonaws.com/324881_09cff2f8816247d5b5750f9983abeb57.html)


