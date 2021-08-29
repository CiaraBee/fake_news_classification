# fake_news_classification
This project explores various machine learning and deep learning approaches to news classification as "real" or "fake", with particular attention being paid to state-of-the-art transformers. The following models are investigated:

Baseline models: 
- Logistic Regression
- Naive Bayes
- Random Forest
- LSTM

BERT-based transformers: 
- BERT
- DistilBERT
- DeBERTa 

Experiments are run to test both in distribution (testing on a holdout test set) and out of distribution (testing on an entirely new dataset with different data distributions) generalisation, showing transformers as the best model types for both. In order to improve generalisation, we propose a two step classification pipeline which identifies and removes opinion based articles from subsequent training data. Opinions are by definition highly subjective and therefore cannot be labelled as "real" or "fake", meaning that these samples in the data may cause models to learn incorrect patterns or patterns that are unique to a dataset labelled by a specific factchecker with a specific belief system, and therefore do not generalise. 

# Data:
The two publicly available datasets used in this project are available for download at:

	ISOT fake news dataset -> https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
	
	Combined Corpus dataset -> https://github.com/JunaedYounusKhan51/FakeNewsDetection/tree/master/Dataset/Combined_Corpus

The manually created fact_opinion dataset is available at: BlackledgeC_200743862_CSC8639_fact_opinion_dataset.csv

# Notebooks:
fact_opinion.ipynb -> This notebook contains the code needed to create the fact/opinion classifier needed for the two step classification pipeline.
		      This notebook imports the manually created fact_opinion dataset CSV file and trains a DistilBERT model for fact/opinion classification
		      The model created can then be saved and used for experiments in the experiments.ipynb notebook.

experiments.ipynb -> This notebook contains the code needed to run all machine learning and deep learning experiments.
		     Data is imported from the publicly available ISOT and Combined Corpus datasets (links for download above).
		     This data is then cleaned to create two dataframes: isot_df and cc_df containing text content and their relevant label (1 for fake news, 0 for real news).
		     This data is then used to train baseline models and three BERT based transformers.
		     These models are trained and tested on holdout testsets (in distribution generalisation) and unseen datasets.
		
# Two step classification pipeline: 
To filter opinion based articles from training data, use the opinion_filtering() function, loading in the trained model from the fact_opinion.ipynb notebook. 
