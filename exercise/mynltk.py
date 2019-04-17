# """
# @Project: gitcode
# @File : mynltk.py
# @Author : yexm
# @E-mail : yexingmin06221@126.com
# @Date : 2019-04-02 17:33:44
# """
# import nltk
# import jieba
# # all_words = nltk.FreqDist(w.lower() for w in nltk.word_tokenize("I'm foolish foolish man"))
# # print(all_words.items())
# # all_words.plot()
# # all_words.plot(2, cumulative=True)
#
# # porter = nltk.PorterStemmer()
# # print(porter.stem('had'))
#
#
#
# print(list(jieba.cut("我是一名码农")))
#
# print(nltk.sent_tokenize("I'm super lying man"))
#
# sent = "I'm super lying man"
# porter2 = nltk.stem.WordNetLemmatizer()
# print([porter2.lemmatize(x) for x in nltk.word_tokenize(sent)])
# print(nltk.pos_tag(sent))
#
# print(nltk.pos_tag("张三来自北京"))
# print( nltk.pos_tag(['love','and','hate']))
#
# tagged_token = nltk.tag.str2tuple('fly/NN')
# print(tagged_token)
#
# sent = '我/NN 是/IN 一个/AT 大/JJ 傻×/NN'
# print([nltk.tag.str2tuple(t) for t in sent.split()]) # 中文语料词性标注（&分词）
#
# print(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize("张三来自北京")) ))
from nltk.book import *
# *** Introductory Examples for the NLTK Book ***
# Loading text1, ..., text9 and sent1, ..., sent9
# Type the name of the text or sentence to view it.
# Type: 'texts()' or 'sents()' to list the materials.
# text1: Moby Dick by Herman Melville 1851
# text2: Sense and Sensibility by Jane Austen 1811
# text3: The Book of Genesis
# text4: Inaugural Address Corpus
# text5: Chat Corpus
# text6: Monty Python and the Holy Grail
# text7: Wall Street Journal
# text8: Personals Corpus
# text9: The Man Who Was Thursday by G . K . Chesterton 1908
print(text1.name)#书名
print(text1.concordance(word="love"))#上下文
print(text1.similar(word="very"))#相似上下文场景
print(text1.common_contexts(words=["pretty","very"]))#相似上下文
text4.dispersion_plot(words=['citizens','freedom','democracy'])#美国总统就职演说词汇分布图
print(text1.collocations())#搭配
print(type(text1))
print(len(text1))#文本长度
print(len(set(text1)))#词汇长度
fword=FreqDist(text1)
print(text1.name)#书名
print(fword)
voc=fword.most_common(50)#频率最高的50个字符
fword.plot(50,cumulative=True)#绘出波形图
print(fword.hapaxes())#低频词