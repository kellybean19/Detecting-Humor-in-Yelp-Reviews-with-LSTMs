# Predicting-Humor-in-Yelp-Reviews

<img src="imgs/funnyreview.png" width="625"/>

## Motivation

Humor detection in texts has many interesting applications. When building artificial intelligence systems such as chatbots and virtual assistants, identifying humor in users' input text can enhance the overall user experience by enabling the system to understand the real motives behind a user's queries. An advanced outcome from this would be to inject humor into computer-generated responses, making human-computer interactions feel more engaging and entertaining.

In the context of review data, funny reviews often draw more attention to the business or product. Identifying these humorous reviews could allow them to be pushed to the top of a page to increase user engagement.

For this project I sought to build my own humor detector. I worked with the Yelp Open Dataset, which contains about 8.6 million reviews, each with a 'funny' attribute corresponding to the number of votes a review has received.


## Defining Humor ##

A look at the histogram for the distribution of the 'funny' column shows a heavy right skew, with the majority of review having 0 votes. The number of votes ranged up to 610.

<img src="imgs/funnyhist_full.png" width="425"/>

After examining individual reviews, 3 votes seemed to be a good cut-off for defining whether a review is humorous to account for noise in 1 and 2 vote reviews. Using this definition, 340,864 out of 8,635,403 total reviews are considered funny. I chose to turn this into a binary classification problem of funny vs not funny, and used undersampling to create a balanced training dataset.

<img src="imgs/funnyvsnotfunnycounts.png" width="425"/>

## Data and Text Processing ##

The data was downloaded from the Yelp website. A preview of the data after removing irrelevant columns can be found in 'data'. Before feeding the text data into the models, I had to get it into the proper format. 

I started with logistic regression to get a baseline classification. My text processing pipeline involved:

- Normalization and encoding
- Tokenization
- Vectorization

I opted to keep stopwords, punctuation, and casing, as I figured that these would contribute to humor. I then vectorized the data into a TF-IDF matrix of 3000 features, including bi-grams. TF-IDF is a word count vectorization takes into account the frequency with which a term appears in a review, as well as how many times it appears in other reviews, to assign a feature a numerical statistic that reflects its relevance to a given review.

For the LSTMs, the process involved:


I chose maximum features of 10000 most frequent words, max sequence length of 200. If a review contains more or less than the max sequence length, it will either be truncated or padded so the input sequences all have the same length for modeling.


## Modeling and Evaluation ##

The four model architectures I compared are:

- Logistic Regression on TF-IDF
- LSTM with one layer
- LSTM with two layers
- Bidirectional LSTM with one layer



Precision 
optimize for recall
(minimize false negatives)