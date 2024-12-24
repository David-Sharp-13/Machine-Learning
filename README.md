# Machine-Learning
Created for Machine Learning 4331 Course. 

My final project was to detect neurological diseases such as Alzheimer's and Frontotemporal Dementia by reading EEG signals from 88 patients.

First, traditional machine learning techniques (SVM RBF, KNN, Random Forest, Gradient Boosting, XGBoost) were used, and their performances were compared using Accuracy, Precision, Recall, and F1 scores.
After this, a deep learning model was created. Because of the time-sequential nature of EEG signals, a long-short-term memory recurrent neural network was also created.

It was found that the RNN performed much better than the traditional models. Additionally, the traditional models were much more computationally expensive and took much longer to process. Therefore, RNNs have the potential to be low-cost, efficient models that can be used to diagnose neurological diseases in patients from their EEG data in real-time.
