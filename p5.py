import numpy as np
import sklearn

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

def test_accuracy(predict, actual):
	cnt = 0
	for i in range(len(actual)):
		if actual[i]!=predict[i]:
			cnt += 1
	return (len(actual) - cnt)/(len(actual))

digits =  load_digits(n_class=2)
train_data, test_data, train_target, test_target = train_test_split(digits.data, digits.target)

lr = LogisticRegression(random_state=0)
lr.fit(train_data, train_target)
test_predict_lr = lr.predict(test_data)
print("the accuracy for Logistic Regression is:", test_accuracy(test_predict_lr, test_target))

rfc = RandomForestClassifier(random_state=0)
rfc.fit(train_data, train_target)
test_predict_rfc = rfc.predict(test_data)
print("the accuracy for Random Forest Classifier is:", test_accuracy(test_predict_rfc, test_target))

svc = SVC(kernel = 'linear', probability = True, C=1E3)#very large C means hard margin
svc.fit(train_data, train_target) 
test_predict_svc = svc.predict(test_data)
print("the accuracy for SVC is:", test_accuracy(test_predict_svc, test_target))

vc = VotingClassifier(estimators=[('lr', lr), ('rf', rfc), ('svc', svc)], voting='soft')
vc.fit(train_data, train_target) 
test_predict_vc = vc.predict(test_data)
print("the accuracy for Voting Classifier is:", test_accuracy(test_predict_vc, test_target))

# After several experiments, we notice that only the random forest classifier 
# shows a tiny rate of misclassification while others remain 100% accurate in our trials.