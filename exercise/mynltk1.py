"""
@Project: gitcode
@File : mynltk1.py
@Author : yexm
@E-mail : yexingmin06221@126.com
@Date : 2019-04-02 18:46:46
@Desc : 根据名字判断性别
"""
from nltk.corpus import names
import  nltk
#特征取的是最后一个字母
def gender_features(word):
    return {'last_letter': word[-1]}
#数据准备
name=[(n,'male') for n in names.words('male.txt')]+[(n,'female') for n in names.words('female.txt')]
print(len(name))
#特征提取和训练模型
features=[(gender_features(n),g) for (n,g) in name]
classifier = nltk.NaiveBayesClassifier.train(features[:6000])
#测试
print(classifier.classify(gender_features('Abby')))
from nltk import classify
print(classify.accuracy(classifier,features[6000:]))

