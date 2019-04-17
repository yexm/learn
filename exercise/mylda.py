"""
@Project: gitcode
@File : mylda.py
@Author : yexm
@E-mail : yexingmin06221@126.com
@Date : 2019-04-06 16:59:44
"""
import jieba
import gensim
def load_stop_words(file_path):
    stop_words = []
    with open(file_path,encoding='utf8') as f:
        for word in f:
            stop_words.append(word.strip())
    return stop_words
def pre_process(data):
    # jieba 分词
    cut_list = list(map(lambda x: '/'.join(jieba.cut(x,cut_all=True)).split('/'), data))
    # 加载停用词 去除 "的 了 啊 "等
    stop_words = load_stop_words('stop_words.txt')
    final_word_list = []
    for cut in cut_list:
        # 去除掉空字符和停用词
        final_word_list.append(list(filter(lambda x: x != '' and x not in stop_words, cut)))
    # print(final_word_list)
    word_count_dict = gensim.corpora.Dictionary(final_word_list)
    # 转成词袋模型 每篇文章由词字典中的序号构成
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in final_word_list]
    # print(bag_of_words_corpus)
    #返回 词袋库 词典
    return bag_of_words_corpus, word_count_dict

def train_lda(bag_of_words_corpus, word_count_dict):
    # 生成lda model
    lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=10, id2word=word_count_dict)
    return lda_model

# 新闻地址 http://news.xinhuanet.com/world/2017-12/08/c_1122082791.htm

train_data = [u"中方对我们的建交国同台湾开展正常经贸和民间往来不持异议，但坚决反对我们的建交国同台湾发生任何形式的官方往来或签署任何带有主权意涵的协定或合作文件",
     u"湾与菲律宾签署了投资保障协定等７项合作文件。菲律宾是台湾推动“新南向”政策中首个和台湾签署投资保障协定的国家。",
     u"中方坚决反对建交国同台湾发生任何形式的官方往来或签署任何带有主权意涵的协定或合作文件，已就此向菲方提出交涉"]
processed_train_data = pre_process(train_data)

lda_model = train_lda(*processed_train_data)
for v in lda_model.print_topics(2,num_words=2):
    print(v)


# import numpy as np
# import lda
#
# X = lda.datasets.load_reuters()
# # print(X.shape)
#
# vocab = lda.datasets.load_reuters_vocab()
# # len(vocab)# 这里是所有的词汇
#
# title = lda.datasets.load_reuters_titles()
# # print(title[:10])
#
#
# model = lda.LDA(n_topics = 20, n_iter = 1500, random_state = 1) #初始化模型, n_iter   迭代次数
# model.fit(X)
#
# topic_word = model.topic_word_
# print(topic_word.shape)
# print(topic_word)
#
# for i, topic_dist in enumerate(topic_word):
#     print(np.array(vocab)[np.argsort(topic_dist)][:-9:-1])

#
# doc_topic = model.doc_topic_
# print(doc_topic.shape)  # 主题分布式395行，20列的矩阵，其中每一行对应一个训练样本在20个主题上的分布
# print("第一个样本的主题分布是",doc_topic[0]) # 打印一下第一个样本的主题分布
# print("第一个样本的最终主题是",doc_topic[0].argmax())
