#Introduction

(Warning!! Under construction! unstable version with a messy code lib)

`Neural Learner for English Language Test (Grammar Complication)` is a final project for the NLP course in Fudan Unviersity. 

###Motivation

I started work on this project because, as far as I am concerned, a hard problem is beneficial to train and obtain word representation/embedding of good quality.

c.f. Zweig, Geoffrey, and Christopher JC Burges. The Microsoft Research sentence completion challenge. Technical Report MSR-TR-2011-129, Microsoft, 2011.



### What is Grammar Complication (语法填空)?
`Grammar Complication` is a type of questions in National College Entrance Examination in China, which mainly aims at testing students' ability on morphology and usage of function words and tenses. 



	  My sister had dropped out of school and made very unwise decisions with her life. She chose to spend her time with people who were lost   _____  she was. They all chose to ignore their  ______ (responsible) and supported one another in a life which involved drinking and partying.  ____ (sad), they were all losing time. They were young and had the potential to become  ______ they wanted if they would only choose to respect themselves and believe in a better life.
	   My sister was lost but my father never gave up on her. She may not have even known it but his prayers and faith ______ her may have been the very thing she needed. I remember sitting at the family dinner table ______ everyone had gathered except my sister. Once again she had chosen to drink with friends instead of spending   an evening with our family who loved ______ very much. We said she would not come. But my father said she would. We all rallied against him, ______ (bet) she would not show up and asked why he would say that. We were convinced he was in denial. He simply ______, “I will always bet on her, on all of you.”

So, a sample question is presented as above, and testers should fill in the blank with or without the cue words.


You could view my project report (written in Chinese) for more detail.


#Build a model

Given a corpus and to predict the missing is a typical and basic task in NLP. It's straight forward to build a statistical language model（LM）.

But, before we dive in the model itself, it might be useful to classify the blank we are going to fill in.

1. morphology of a given cue word
2. function words
	
	* determiners (the, that, ...),
	* conjunctions (and, but, so, since, as...),
	* prepositions (in, of, at, through, over, between, under, ...),
	* pronouns (she, they, ...),
	* auxiliary verb (be, have, do, has, will, has been, did, ....),
	* modals (may, could...),
	* quantifiers (some, both...),
	* etc.

The first kind question can be solved if we enumerate all the different tenses of a given word, hence it is a classification problem.

According to the manual of the exam, there're roughly 100 function word included in the exam. So, the second kind of words can be regarded as a classification problem whose size of categories is 100.


#Solution

Since it is reduced as a classification problem, I have tried several classifiers: Naive-Bayes classifier, Max-Ent classifier and random forest classifier. Futhermore, I have tried MLP neural network and LSTM/GRU (without tuning), performance of which is better than traditional classifiers. A better result is supposed to show with a good tuning and/or other sophisticated neural network architecture (e.g. with hierachical memory, external memory).

#Result

|| Naive-Bayes        | Max-Ent           | Random Forest  | MLP| LSTM (L2 Norm)| |
| ------------- |:-------------:| -----:|:-------------:| -----:|
|feature_0      | 28.3 | 32.2 | 10.0|8.7|36|window size = 3|
|feature_1      | 30.4	|35.3	|19.2	|19.3	|41.3|	windows = 5|
|feature_2| 36.0	|34.2|	22.3	|17.2|	40	|windows = 7|



