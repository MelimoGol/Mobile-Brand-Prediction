# Mobile-Brand-Prediction
The problem is predicting mobile phone brands in advertisements posted on Divar. In this project, the Divar dataset that consists of one million advertisements was used. First, mobile ads were extracted. First, mobile ads were extracted, and a bag of words was made. After checking the columns of the dataset, I realized that the desc and title columns are helpful in predicting the mobile brand. So I concatenated them and tokenized the result using the NLTK's word_tokenize. Since these columns have Persian and English sentences and words, I pre-processed their tokens with the Hazm and NLTK libraries. Then stemming were done, the stop words were removed, and the English and Persian tokens were combined and considered as features.
Next the Naïve Bayes and Logistic  Regression classifiers were trained to predict the mobile brand. Finally, the fold-K method was used to validate models.

# Dataset
Cafebazaar Research Group published a dataset consisting of advertisements posted on the Divar application, which you can download from this link:
https://research.cafebazaar.ir/visage/divar_datasets/

To predict the mobile brand, the desc and title columns of the dataset are helpful.
# What-is-Divar
https://en.wikipedia.org/wiki/Divar_(website)
