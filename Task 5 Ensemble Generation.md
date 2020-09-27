## Ensemble Generation methods
# 1. 平均法

1.1 简单平均法
- 简单加权平均，结果直接融合 求多个预测结果的平均值。pre1-pren分别是n组模型预测出来的结果，将其进行加权融。

      #公式：
      pre = (pre1 + pre2 + pre3 +...+pren )/n
      
      #生成一些简单的样本数据， test_prei代表第i个模型的预测值
      test_pre1 = [1.2, 3.2, 2.1, 6.2]
      test_pre2 = [0.9, 3.1, 2.0, 5.9]
      test_pre3 = [1.1, 2.9, 2.2, 6.0]
      
      # y_test_true 代表模型的真实值
      y_test_true = [1, 3, 2, 6]
      
      def Mean_method(test_pre1,test_pre2,test_pre3):
          Mean_result = pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).mean(axis=1)
          return Mean_result
          
       test_pre_mean = Mean_method(test_pre1,test_pre2,test_pre3)
       
       print('Pred_mean MAE:',mean_absolute_error(y_test_true, test_pre_mean)) 
       Pred_mean MAE: 0.06666666666666693   
          
1.2 加权平均法
- 加权平均法 一般根据之前预测模型的准确率，进行加权融合，将准确性高的模型赋予更高的权重。

      #公式：
      pre = 0.3pre1 + 0.3pre2 + 0.4pre3
      
      def Weighted_method(test_pre1, test_pre2, test_pre3, w=[1/3, 1/3, 1/3]):
          Weighted_result = w[0] * pd.Series(test_pre1) + w[1] * pd.Series(test_pre2) + w[2] * pd.Series(test_pre3)
          return Weighted_result
      
      # 分别计算的pre1，pre2，pre3的各自MAE，然后再根据3个MAE计算每个模型的权重, 计算方式就是： wi = mae(i) / sum(mae)
      w = [0.3, 0.4, 0.3]
      
      weighted_pre = Weighted_method(test_pre1, test_pre2, test_pre3, w)
      
      print('Pred_Weight MAE:',mean_absolute_error(y_test_true, weighted_pre)) 
      Pred_Weight MAE: 0.05750000000000027
      
可以看出Pred_mean MAE: 0.067 > Pred_Weight MAE: 0.058。由此，加权平均法的效果略好于简单平均法。加权融合一般适用于回归任务中模型的结果层面，而对于分类模型，假设存在多个不同的模型，多个模型具有不同的分类结果。对于一个对象而言，最终的分类结果可以采用投票最多的类为最终的预测结果。
      
      
      
# 2. 投票法

2.1 简单投票
- 基本思想是选择所有机器学习算法当中输出最多的那个类，少数服从多数。又叫硬投票(Majority/Hard voting)。

      from xgboost import XGBClassifier
      from sklearn.linear_model import LogisticRegression
      from sklearn.ensemble import RandomForestClassifier, VotingClassifier
      clf1 = LogisticRegression(random_state=1)
      clf2 = RandomForestClassifier(random_state=1)
      clf3 = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=4, min_child_weight=2, subsample=0.7,objective='binary:logistic')
 
      vclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('xgb', clf3)])
      vclf = vclf .fit(x_train,y_train)
      print(vclf .predict(x_test))

举例说明：

      模型1 A-99% B-1%
      模型2 A-49% B-51%
      模型3 A-40% B-60%
      模型4 A-90% B-10%
      模型5 A-30% B-70%
      
      #根据少数服从多数原则，最终投票结果为B（3票）
虽然只有模型1和模型4分类结果为A，但概率都高于90%，也就是说很确定结果为A，其他三个模型分类结果是B，但从概率看，并不是很确定。这时如果用hard voting得出的最终结果为就显得不太合理。

2.2 加权投票
- 有时候少数服从多数并不适用，那么更加合理的投票方式，应该是有权值的投票方式，又叫软投票(Soft voting)。使用各个算法输出的类概率来进行类的选择，输入权重的话，会得到每个类的类概率的加权平均值，值大的类会被选择。在VotingClassifier中加入参数 voting='soft', weights=[2, 1, 1]，weights用于调节基模型的权重。

      vclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('xgb', clf3)], voting='soft', weights=[2, 1, 1])
      vclf = vclf .fit(x_train,y_train)
      print(vclf .predict(x_test))
      
举例说明：

      模型1 A-99% B-1%
      模型2 A-49% B-51%
      模型3 A-40% B-60%
      模型4 A-90% B-10%
      模型5 A-30% B-70%
      
      #把概率当成权重，计算每个类的类概率的加权平均值，值大的类会被选择。wi = p(i) / sum(i)。P(A)=0.616，P(B)=0.384，最终投票结果为A
      
# 3. Stacking
stacking 将若干基学习器获得的预测结果，将预测结果作为新的训练集来训练一个学习器。如下图 假设有五个基学习器，将数据带入五基学习器中得到预测结果，再带入模型六中进行训练预测。但是由于直接由五个基学习器获得结果直接带入模型六中，容易导致过拟合。所以在使用五个及模型进行预测的时候，可以考虑使用K折验证，防止过拟合。

- 首先我们将训练集分成5份（5折交叉验证）
- 对于每一个基模型i来说， 我们用其中的四份进行训练， 然后用另一份训练集作为验证集进行预测得到Pi的一部分， 然后再用测试集进行预测得到Ti的一部分，这样当五轮下来之后，验证集的预测值就会拼接成一个完整的Pi, 测试集的label值取个平均就会得到一个Ti。
- 这些Pi进行合并就会得到下一层的训练集train2, Ti进行合并就得到了下一层的测试集test2。
- 利用train2训练第二层的模型， 然后再test2上得到预测结果，就是最终的结果。
 ![stacking](https://camo.githubusercontent.com/8208e1d00c405d9ca57dbd557a6a580e4d9781df/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f323032303039313330313234303235362e706e67) 
 
# 4. Blending
blending 与stacking不同，blending是将预测的值作为新的特征和原特征合并，构成新的特征值，用于预测。为了防止过拟合，将数据分为两部分d1、d2，使用d1的数据作为训练集，d2数据作为测试集。预测得到的数据作为新特征使用d2的数据作为训练集结合新特征，预测测试集结果。

  ![stacking](https://camo.githubusercontent.com/a8ef2ee92fe0a5dc3b00dee61b6559f25f085f6b/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031323430313935382e706e67)
  
  
# More for reading
[Understanding ensemble generation](https://blog.csdn.net/wuzhongqiang/article/details/105012739)

[Stack and blend with models](https://tianchi.aliyun.com/notebook-ai/detail?postId=131986)
