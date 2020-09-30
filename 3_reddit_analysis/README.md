# Analysis of Reddit posts relating to climate change

## Getting started:
1. Run `pip install -r requirements.txt` to get all packages.
2. Update `config.json` so that the path for `base_dir` is the location of this subdirectory (so it will end in `/3_reddit_analysis`). For the other paths, you can leave them as is if you are using this repository on NLP cluster, otherwise, you should `scp` the data from those locations to somewhere you'll be able to read them from, and update their paths accordingly. 
3. Download the file `posts_with_words.pkl.zip` (containing all of the Reddit climate change data) from [here](https://drive.google.com/file/d/1z29MgH2WGN0JN8R5r07r6MUV1CCpzKF0/view?usp=sharing) and unzip it to the `reddit_data` subdirectory.

Now you can run `python regress_reddit.py` to go through the entire multiple linear regression pipeline on the default feature sets, dependent variables, and subsets of the Reddit data. The coefficients etc. will be saved to a dataframe and also plotted, with results saved to `posts_with_words.pkl_out`.

There are more detailed explanations of the different parts of `regress_reddit.py` in `demo.ipynb`.

You can also use the scripts in `utils.py` to explore correlations among features and visualize more fine-grained distributions (e.g., of a feature broken down by subreddit). These are also illustrated in `demo.ipynb`.

## Other notes:
* This repository was made using python==3.6.8.
* When running on the NLP cluster, I've been setting the memory flag to 100GB (lower may also be fine but I haven't tried).
* As explained in `demo.ipynb`, cached feature files with pre-computed features for `posts_with_words.pkl` (found in `/posts_with_words.pkl_out/cached`) are used in the featurization step to speed things up. If you want to compute these features from scratch, you should move them to a different location. Warning: computing the SentiStrength scores takes a *long* time (~40 hours).
