# Predicting-Humor-in-Yelp-Reviews

<img src="imgs/funnyreview.png" width="425"/>

## Motivation

Humor is a very human quality and skill, and it’s something we have all experienced at some point. It's also a very subjective thing, but what if there was a way to systematically detect it using data that is readily available to us? To answer this question, I sought to detect humor in the community-based context of Yelp. For this goal I worked with the Yelp Open Dataset, which contains ~8.6 million reviews of businesses on Yelp, each with a 'funny' attribute representing the number of funny votes a given review has received.


This project beings by creating baseline models to predict humor in Yelp Reviews using Natural Language processing and classic machine learning methods. Useful applications include helping business owners better understand user engagement with their business. It can also signal to other users about the exaggerated nature of humorous reviews, and that such reviews should be taken with a grain of salt.

Perhaps most interestingly, there is potential to incorporate humor in AI systems to make human-computer interactions feel more human.


## Defining Humor ##

A look at the histogram for the distribution of the 'funny' column (up to 50 votes) shows a heavy left skew, with the majority of review having 0 votes. The number of votes ranged up to 610.

<img src="imgs/funnyhist.png" width="425"/>

After examining individual reviews, 3 votes seemed to be a good cut-off for defining whether a review is humorous to account for noise in 1 and 2 vote reviews. Using this definition, only 340,864 out of 8,635,403 total reviews are considered funny. I chose to turn this into a binary classification problem of funny vs not funny, and used undersampling to create a balanced dataset of 200k reviews total, 100k each of funny/not funny, to make model results interpretable.

## Data and Text Processing ##

The data was downloaded from the Yelp website. A preview of the data after removing irrelevant columns can be found in 'data'. 

My text processing pipeline involved experimenting with the process below:

- Normalization and encoding
- Tokenization
- Lowercasing
- Removing stopwords
- Removing punctuation
- Lemmatization

I opted to keep stopwords, punctuation, and casing, as I figured that these would contribute to humor. I also experimented with lemmatization and found that it did not improve model performance, so I decided to keep the original words.

I then vectorized the data into a TF-IDF matrix of 3000 features, including bi-grams and tri-grams. TF-IDF takes into account the frequency with which a term appears in a review, as well as how many times it appears in other reviews, to assign a feature a numerical statistic that reflects its relevance to given review.


## Modeling and Evaluation ##

Once I had the data ready, I chose to experiment with a variety of classification models, including:

- Logistic Regression
- Naive Bayes
- Random Forest
- Gradient Boosting

which were all done on top of the TF-IDF vectorization of bag-of-words.

Logistic Regression and Naive Bayes took the shortest amount of time, while Gradient Boosting took the longest.

The results show that logistic regression performed the best. I chose to use ROC curves and the AUC to evaluate my models, as this gives the percentage likelihood that a model will correctly predict whether a review is funny or not.

<img src="imgs/roccurves.png" width="625"/>

- Logistic Regression AUC: 0.838 
- Naive Bayes AUC: 0.796
- Random Forest AUC: 0.775
- Gradient Boosting AUC: 0.827

The more complex models did not perform as well as Logistic Regression, although they all did much better than the baseline of pure chance.

## Model Interpretation ##

Once I had my winning model, I took a look at the feature importance of my 3000 keywords to try to gain an understanding of the words that contributed the most to humor.


<img src="imgs/featureimportance.png" width="625"/>

As it turns out, most of the top features were fairly neutral words. There were some spicier words like 'drunk' and 'dude', but overall the words by themselves aren’t very funny.

This makes sense as reviews are composed of sentences and combination of keywords, so it’s hard to see from the words themselves whether something is funny.

In order to get better representations of humor, I would need to incorporate semantic analysis, which I plan on doing as next steps.

Precision 
optimize for recall
(minimize false negatives)