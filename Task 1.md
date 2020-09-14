A non-performing loan is a loan that is in default or close to being in default. The borrower hasn't made any scheduled payments of principal or interest for some time.

## Datasets
The datasets include one train.csv and one testA.csv. 

    import pandas as pd
    train = pd.read_csv('train.csv')
    testA = pd.read_csv('testA.csv')
    print('Train data shape:',train.shape) 
    (800000, 47)
    print('TestA data shape:',testA.shape)
    (200000, 48)

## Metrics
AUC - area under curve
1.  Confusion Matrix

    ![confusion-matrix](https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg)  
2. Accuracy = (True Positive + True Negative) / (True Positive + True Negative + False Positive + False Negative)
3. Precision = True Positive / (True Positive + False Positive)
4. Recall = True Positive / (True Positive + False Negative)
5. F1-score = 2 / (1/precision+1/recall)

  Metrix codes

    #confusion matrix
    import numpy as np
    from sklearn.metrics import confusion_matrix
    y_pred = [0, 1, 0, 1]
    y_true = [0, 1, 1, 0]
    print('混淆矩阵:\n',confusion_matrix(y_true, y_pred))
    
    #accuracy
    from sklearn.metrics import accuracy_score
    y_pred = [0, 1, 0, 1]
    y_true = [0, 1, 1, 0]
    print('ACC:',accuracy_score(y_true, y_pred))
    
    #Precision,Recall,F1-score
    from sklearn import metrics
    y_pred = [0, 1, 0, 1]
    y_true = [0, 1, 1, 0]
    print('Precision',metrics.precision_score(y_true, y_pred))
    print('Recall',metrics.recall_score(y_true, y_pred))
    print('F1-score:',metrics.f1_score(y_true, y_pred))
    
6. Precision-Recall Curve: shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.

![PRC](https://camo.githubusercontent.com/25688baabbff569136e19f04c812ed80af778351/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031303232363132352e706e67)

7. Receiver Operating Characteristic: The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
* TPR = TP / TP + FN
* FPR = FP / FP + TN

![ROC](https://camo.githubusercontent.com/2dfea351fa5eac42caab9e716aa76a20553e4103/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031303232363132342e706e67)

8. Area Under Curve: area below ROC. As usual, the curve is above the line y=x.Thus, the range of AUC is 0.5-1. 
ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes.
When AUC is close to to 1, it is perfectly able to distinguish between positive class and negative class, while AUC is close to 0.5, model has no discrimination capacity to distinguish between positive class and negative class.

  Curve codes
  
    #P-R cureve
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot(precision, recall)
    
    # ROC
    from sklearn.metrics import roc_curve
    y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    FPR,TPR,thresholds=roc_curve(y_true, y_pred)
    plt.title('ROC')
    plt.plot(FPR, TPR,'b')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    
    # AUC score
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    print('AUC socre:',roc_auc_score(y_true, y_scores))
    
    # KS值 在实际操作时往往使用ROC曲线配合求出KS值
    from sklearn.metrics import roc_curve
    y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
    FPR,TPR,thresholds=roc_curve(y_true, y_pred)
    KS=abs(FPR-TPR).max()
    print('KS值：',KS)

# More for reading
[Understanding AUC - ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

