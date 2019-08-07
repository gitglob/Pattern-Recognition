# project for ECE-345 (Pattern Recognition)
_Sentiment Analysis and Prediction on Twitter Reviews dataset of Airline Companies_ (**_Python_**)

What **dataset** did we have? 
* A really big dataset with tweets and other information of twitter users regarding their flights with certain airline companies.

What was our **task**?
* To use whatever techniques we deemed fit, in order to :
  1. Extract information from the tweets and the other columns
  2. Provide insight about the quality of the airline's services, along with useful visualizations
  3. Check the performance of the techniques used and compare
    
What **type of problem** was it?
* It was a classification problem, since we had a target column which stated the customer's sentiment, 
  but it could easily become a clustering problem, had we ignored the target column.
    
What **techniques** did we use?
  1. Preprocessing required Natural Language Processing and Word Embedding, among with typical problems like handling NaN values...
  2. Classic Machine Learning algorithms (Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Naive Bayes, Gradient Boosting
  3. Neural Networks (Convolutional Neural Networks, Simple Keras Classifier with embedding layer, LSTM)
  
  
  
Important to **note**:
* We used and tested k-folds evaluation, train/test splitting, multiple optimizers, GRID search, 
    different learning rates, and multiple # of hidden layers, and tried to get the best possible results. 
