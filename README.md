# cs229_reacclimate

Project outline:
* A. Identifying words correlated with high engagement
	1. Getting data w/ signals of engagement
		- [ ] Find appropriate APIs and request parameters for:
			- [ ] tweets about climate change etc.
			- [ ] posts from subreddits about climate change etc.
	2. Running Reid's deconfounded lexicon induction model
		- [ ] decide on potential confounding variables
* B. Identifying effective words
	1. Review literature on effective climate change communication to compile list of frames/strategies
		- [ ] framing literature (economic cost, (scientific) uncertainty)
		- [ ] Gabrielle Wong-Parodi
	2. Operationalize effective strategies computationally
		- [ ] create/curate lexicons associated with given frames/strategies 
	3. Train word embedding models
		- [ ] gather datasets (news articles, tweets, Reddit posts)
		- [ ] separate datasets according to variables like attitude, audience, author
	4. Query models for replacement terms
* Packaging both components in easy-to-use interface

Potential project timeline:
- 1. Data collection and pre-processing for components A. and B. (2 weeks; July 13-July 27)
- 2. Component A. (run Reid's code) (1 week; July 27-August 3)
- 3. Component B. 
	- i. Lit review (up to 3 weeks; July 13-August 3)
	- ii. Create some seed lexicons (1 week; August 3-August 10)
	- iii. Train and query VSMs (3 weeks; August 11-August 31)
- 4. Create model interfaces (September)

Old:
- [x] Reach out to Allison Koenecke about using off-the-shelf tweet classifier
- [x] Classify Harold et al's tweet data
- [x] Retrain language models for climate change affirmers, non-affirmers
- [ ] Deconfounded lexicon induction for engagement prediction

Ideas for TODOs:

- [ ] Context-aware engagement prediction
  - [ ] Deconfounded lexicon induction (https://nlp.stanford.edu/pubs/pryzant2018lexicon.pdf)
  - [ ] Weight engagement function by some kind of similarity function between target and original context(s)
- [ ] Stance awareness (target = "Climate change is a real concern")
  - [ ] Intuition: want to maximize engagement score for tweets with pro-target stance and minimize engagement score for tweets with anti-target stance
  - [ ] Build classifier to classify stance of a tweet (based on: text features, account following network)
  - [ ] Training data: SemEval 2016 Task 6?
- [ ] Other algorithm tweaks
  - [ ] Thesaurus sources beyond WordNet to generate alternative candidates
  - [ ] Making use of language models: learn embedding space for each side (anti-/pro- climate change being a real concern) and propose nearest neighbors of positive sentiment words ("inspiration", "awesome", "economic growth") as candidates
  - [ ] Or weight engagement function by cosine similarity between candidate word and set of positive sentiment words (or some other LIWC category--social processes, achievement, etc.) 
- [ ] Create evaluation metrics
  - [ ] Test data: manually annotate a set of tweet-length sentence pairs (differing in a candidate word/phrase) for their relative improvement
  - [ ] Look into other sources

References:
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


