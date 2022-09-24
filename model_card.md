# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Jhonsen Djajamuliadi created the model in Sept 2022. The model is a random forest classifer, trained using default hyperparameters with scikit-learn v.1.1.2.  
## Intended Use
This model should be used to predict the household income based on answers to census questionares. The users are policy makers and economists.  
## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The target class was binary, i.e., income `<=50k` and `>50k` 

The original data set has 1728 rows, and a 90-10 split was used to break this into a train and test set. No stratification was done.  To use the data for training a One Hot Encoder was used on the features. To handle class imbalance, [SMOTENN](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html) was used to oversample the minority class.  

## Metrics
The model was evaluated using accuracy and other classification metrics.
```
recall   : 0.6722797927461139
precision: 0.64875
accuracy : 0.8347260909935005
f1-score : 0.6603053435114504
```  

## Ethical Considerations
Although the data has been de-identified, there are information that bears certain ethical consideration, e.g., `native origin` and `race`.  
  
## Caveats and Recommendations
Future work should explore the removal of, e.g., `sex` and `race` from the feature set, to avoid bias.   