# Mobile-Brand-Prediction
The problem is predicting mobile phone brands in advertisements posted on Divar. In this project, the Divar dataset that consists of one million advertisements was used. First, mobile ads were extracted, consisting of 62,176 mobile advertisements. After checking the columns of the dataset, I realized that the desc and title columns are helpful in predicting the mobile brand. So I concatenated them and tokenized the result using the NLTK's word_tokenize. Since these columns have Persian and English sentences and words, I pre-processed their tokens with the Hazm and NLTK libraries. Then stemming was done, the stop words were removed, and the English and Persian tokens were combined and considered as features. I used scikit-learn to make a bag of words from the vector of tokens. But the tokens repeated less than two times or more than 60% were removed because they probably do not provide any information that can be learned and unnecessarily increase the feature vector's size. The final bag of words has 19601 features.

Then, I assigned a numerical value to each brand. For example, '1' was assigned to the Samsung brand and '2' to the Huawei brand. Thus, each data contains a vector of the features and its target value. Finally, using the k-fold validation method with fold = 4, the Naïve Bayes and Logistic Regression classifiers were trained.





# Dataset
Cafebazaar Research Group published a dataset consisting of advertisements posted on the Divar application, which you can download from this link:
https://research.cafebazaar.ir/visage/divar_datasets/

To predict the mobile brand, the desc and title columns of the dataset are helpful.
# What-is-Divar
https://en.wikipedia.org/wiki/Divar_(website)
