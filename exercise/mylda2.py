# -*- coding:utf-8 -*-
import logging
import logging.config
import ConfigParser
import numpy as np
import random
import codecs
import os

from collections import OrderedDict

# 获取当前路径
path = os.getcwd()
# 导入日志配置文件
logging.config.fileConfig("logging.conf")
# 创建日志对象
logger = logging.getLogger()
# loggerInfo = logging.getLogger("TimeInfoLogger")
# Consolelogger = logging.getLogger("ConsoleLogger")

# 导入配置文件
conf = ConfigParser.ConfigParser()
conf.read("setting.conf")
# 文件路径
trainfile = os.path.join(path, os.path.normpath(conf.get("filepath", "trainfile")))
wordidmapfile = os.path.join(path, os.path.normpath(conf.get("filepath", "wordidmapfile")))
thetafile = os.path.join(path, os.path.normpath(conf.get("filepath", "thetafile")))
phifile = os.path.join(path, os.path.normpath(conf.get("filepath", "phifile")))
paramfile = os.path.join(path, os.path.normpath(conf.get("filepath", "paramfile")))
topNfile = os.path.join(path, os.path.normpath(conf.get("filepath", "topNfile")))
tassginfile = os.path.join(path, os.path.normpath(conf.get("filepath", "tassginfile")))
# 模型初始参数
K = int(conf.get("model_args", "K"))
alpha = float(conf.get("model_args", "alpha"))
beta = float(conf.get("model_args", "beta"))
iter_times = int(conf.get("model_args", "iter_times"))
top_words_num = int(conf.get("model_args", "top_words_num"))


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0
    # 把整个文档及真的单词构成vocabulary（不允许重复）


class DataPreProcessing(object):
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        # 保存每个文档d的信息(单词序列，以及length)
        self.docs = []
        # 建立vocabulary表，照片文档的单词
        self.word2id = OrderedDict()

    def cachewordidmap(self):
        with codecs.open(wordidmapfile, 'w', 'utf-8') as f:
            for word, id in self.word2id.items():
                f.write(word + "\t" + str(id) + "\n")


class LDAModel(object):
    def __init__(self, dpre):
        self.dpre = dpre  # 获取预处理参数
        #
        # 模型参数
        # 聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta)
        #
        self.K = K
        self.beta = beta
        self.alpha = alpha
        self.iter_times = iter_times
        self.top_words_num = top_words_num
        #
        # 文件变量
        # 分好词的文件trainfile
        # 词对应id文件wordidmapfile
        # 文章-主题分布文件thetafile
        # 词-主题分布文件phifile
        # 每个主题topN词文件topNfile
        # 最后分派结果文件tassginfile
        # 模型训练选择的参数文件paramfile
        #
        self.wordidmapfile = wordidmapfile
        self.trainfile = trainfile
        self.thetafile = thetafile
        self.phifile = phifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile
        # p,概率向量 double类型，存储采样的临时变量
        # nw,词word在主题topic上的分布
        # nwsum,每各topic的词的总数
        # nd,每个doc中各个topic的词的总数
        # ndsum,每各doc中词的总数
        self.p = np.zeros(self.K)
        # nw,词word在主题topic上的分布
        self.nw = np.zeros((self.dpre.words_count, self.K), dtype="int")
        # nwsum,每各topic的词的总数
        self.nwsum = np.zeros(self.K, dtype="int")
        # nd,每个doc中各个topic的词的总数
        self.nd = np.zeros((self.dpre.docs_count, self.K), dtype="int")
        # ndsum,每各doc中词的总数
        self.ndsum = np.zeros(dpre.docs_count, dtype="int")
        self.Z = np.array(
            [[0 for y in xrange(dpre.docs[x].length)] for x in xrange(dpre.docs_count)])  # M*doc.size()，文档中词的主题分布

        # 随机先分配类型，为每个文档中的各个单词分配主题
        for x in xrange(len(self.Z)):
            self.ndsum[x] = self.dpre.docs[x].length
            for y in xrange(self.dpre.docs[x].length):
                topic = random.randint(0, self.K - 1)  # 随机取一个主题
                self.Z[x][y] = topic  # 文档中词的主题分布
                self.nw[self.dpre.docs[x].words[y]][topic] += 1
                self.nd[x][topic] += 1
                self.nwsum[topic] += 1

        self.theta = np.array([[0.0 for y in xrange(self.K)] for x in xrange(self.dpre.docs_count)])
        self.phi = np.array([[0.0 for y in xrange(self.dpre.words_count)] for x in xrange(self.K)])

    def sampling(self, i, j):
        # 换主题
        topic = self.Z[i][j]
        # 只是单词的编号，都是从0开始word就是等于j
        word = self.dpre.docs[i].words[j]
        # if word==j:
        #    print 'true'
        self.nw[word][topic] -= 1
        self.nd[i][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[i] -= 1

        Vbeta = self.dpre.words_count * self.beta
        Kalpha = self.K * self.alpha
        self.p = (self.nw[word] + self.beta) / (self.nwsum + Vbeta) * \
                 (self.nd[i] + self.alpha) / (self.ndsum[i] + Kalpha)

        # 随机更新主题的吗
        # for k in xrange(1,self.K):
        #     self.p[k] += self.p[k-1]
        # u = random.uniform(0,self.p[self.K-1])
        # for topic in xrange(self.K):
        #     if self.p[topic]>u:
        #         break

        # 按这个更新主题更好理解，这个效果还不错
        p = np.squeeze(np.asarray(self.p / np.sum(self.p)))
        topic = np.argmax(np.random.multinomial(1, p))

        self.nw[word][topic] += 1
        self.nwsum[topic] += 1
        self.nd[i][topic] += 1
        self.ndsum[i] += 1
        return topic

    def est(self):
        # Consolelogger.info(u"迭代次数为%s 次" % self.iter_times)
        for x in xrange(self.iter_times):
            for i in xrange(self.dpre.docs_count):
                for j in xrange(self.dpre.docs[i].length):
                    topic = self.sampling(i, j)
                    self.Z[i][j] = topic
        logger.info(u"迭代完成。")
        logger.debug(u"计算文章-主题分布")
        self._theta()
        logger.debug(u"计算词-主题分布")
        self._phi()
        logger.debug(u"保存模型")
        self.save()

    def _theta(self):
        for i in xrange(self.dpre.docs_count):  # 遍历文档的个数词
            self.theta[i] = (self.nd[i] + self.alpha) / (self.ndsum[i] + self.K * self.alpha)

    def _phi(self):
        for i in xrange(self.K):
            self.phi[i] = (self.nw.T[i] + self.beta) / (self.nwsum[i] + self.dpre.words_count * self.beta)

    def save(self):
        # 保存theta文章-主题分布
        logger.info(u"文章-主题分布已保存到%s" % self.thetafile)
        with codecs.open(self.thetafile, 'w') as f:
            for x in xrange(self.dpre.docs_count):
                for y in xrange(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')
                # 保存phi词-主题分布
        logger.info(u"词-主题分布已保存到%s" % self.phifile)
        with codecs.open(self.phifile, 'w') as f:
            for x in xrange(self.K):
                for y in xrange(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')
                # 保存参数设置
        logger.info(u"参数设置已保存到%s" % self.paramfile)
        with codecs.open(self.paramfile, 'w', 'utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')
            # 保存每个主题topic的词
        logger.info(u"主题topN词已保存到%s" % self.topNfile)

        with codecs.open(self.topNfile, 'w', 'utf-8') as f:
            self.top_words_num = min(self.top_words_num, self.dpre.words_count)
            for x in xrange(self.K):
                f.write(u'第' + str(x) + u'类：' + '\n')
                twords = []
                twords = [(n, self.phi[x][n]) for n in xrange(self.dpre.words_count)]
                twords.sort(key=lambda i: i[1], reverse=True)
                for y in xrange(self.top_words_num):
                    word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write('\t' * 2 + word + '\t' + str(twords[y][1]) + '\n')
                    # 保存最后退出时，文章的词分派的主题的结果
        logger.info(u"文章-词-主题分派结果已保存到%s" % self.tassginfile)
        with codecs.open(self.tassginfile, 'w') as f:
            for x in xrange(self.dpre.docs_count):
                for y in xrange(self.dpre.docs[x].length):
                    f.write(str(self.dpre.docs[x].words[y]) + ':' + str(self.Z[x][y]) + '\t')
                f.write('\n')
        logger.info(u"模型训练完成。")
    # 数据预处理，即：生成d（）单词序列，以及词汇表


def preprocessing():
    logger.info(u'载入数据......')
    with codecs.open(trainfile, 'r', 'utf-8') as f:
        docs = f.readlines()
    logger.debug(u"载入完成,准备生成字典对象和统计文本数据...")
    # 大的文档集
    dpre = DataPreProcessing()
    items_idx = 0
    for line in docs:
        if line != "":
            tmp = line.strip().split()
            # 生成一个文档对象：包含单词序列（w1,w2,w3,,,,,wn）可以重复的
            doc = Document()
            for item in tmp:
                if dpre.word2id.has_key(item):  # 已有的话，只是当前文档追加
                    doc.words.append(dpre.word2id[item])
                else:  # 没有的话，要更新vocabulary中的单词词典及wordidmap
                    dpre.word2id[item] = items_idx
                    doc.words.append(items_idx)
                    items_idx += 1
            doc.length = len(tmp)
            dpre.docs.append(doc)
        else:
            pass
    dpre.docs_count = len(dpre.docs)  # 文档数
    dpre.words_count = len(dpre.word2id)  # 词汇数
    logger.info(u"共有%s个文档" % dpre.docs_count)
    dpre.cachewordidmap()
    logger.info(u"词与序号对应关系已保存到%s" % wordidmapfile)
    return dpre


def run():
    # 处理文档集，及计算文档数，以及vocabulary词的总个数，以及每个文档的单词序列
    dpre = preprocessing()
    lda = LDAModel(dpre)
    lda.est()


if __name__ == '__main__':
    run()
# """
# @Project: gitcode
# @File : mylda2.py
# @Author : yexm
# @E-mail : yexingmin06221@126.com
# @Date : 2019-04-17 11:10:04
# """
# #-*- coding:utf-8 -*-
#
# import random
# import codecs
# import os
# import numpy as np
# from collections import OrderedDict
# import sys
#
# print(sys.stdin.encoding)
#
# train_file = 'train_file.txt'
# bag_word_file = 'word2id.txt'
#
# # save file
# # doc-topic
# phi_file = 'phi_file.txt'
# # word-topic
# theta_file = 'theta_file.txt'
#
# ############################
# alpha = 0.1
# beta = 0.1
# topic_num = 10
# iter_times = 100
#
# ##########################
# class Document(object):
#     def __init__(self):
#         self.words = []
#         self.length = 0
#
# class DataDict(object):
#     def __init__(self):
#         self.docs_count = 0
#         self.words_count = 0
#         self.docs = []
#         self.word2id = OrderedDict()
#
#     def add_word(self, word):
#         if word not in self.word2id:
#             self.word2id[word] = self.words_count
#             self.words_count += 1
#
#         return self.word2id[word]
#
#     def add_doc(self, doc):
#         self.docs.append(doc)
#         self.docs_count += 1
#
#     def save_word2id(self, file):
#         with codecs.open(file, 'w','utf-8') as f:
#             for word,id in self.word2id.items():
#                 f.write(word +"\t"+str(id)+"\n")
#
# class DataClean(object):
#
#     def __init__(self, train_file):
#         self.train_file = train_file
#         self.data_dict = DataDict()
#
#     '''
#         input: text-word matrix
#     '''
#     def process_each_doc(self):
#         for text in self.texts:
#             doc = Document()
#             for word in text:
#                 word_id = self.data_dict.add_word(word)
#                 doc.words.append(word_id)
#             doc.length = len(doc.words)
#             self.data_dict.add_doc(doc)
#
#     def clean(self):
#         with codecs.open(self.train_file, 'r','utf-8') as f:
#             self.texts = f.readlines()
#
#         self.texts = list(map(lambda x: x.strip().split(), self.texts))
#         assert type(self.texts[0]) == list , 'wrong data format, texts should be two dimension'
#         self.process_each_doc()
#
# class LDAModel(object):
#
#     def __init__(self, data_dict):
#
#         self.data_dict = data_dict
#         #
#         # 模型参数
#         # 主题数topic_num
#         # 迭代次数iter_times,
#         # 每个类特征词个数top_words_num
#         # 超参数alpha beta
#         #
#         self.beta = beta
#         self.alpha = alpha
#         self.topic_num = topic_num
#         self.iter_times = iter_times
#
#         # p,概率向量 临时变量
#         self.p = np.zeros(self.topic_num)
#
#         # word-topic_num: word-topic matrix 一个word在不同topic的数量
#         # topic_word_sum: 每个topic包含的word数量
#         # doc_topic_num: doc-topic matrix 一篇文档在不同topic的数量
#         # doc_word_sum: 每篇文档的词数
#         self.word_topic_num = np.zeros((self.data_dict.words_count, \
#             self.topic_num),dtype="int")
#         self.topic_word_sum = np.zeros(self.topic_num,dtype="int")
#         self.doc_topic_num = np.zeros((self.data_dict.docs_count, \
#             self.topic_num),dtype="int")
#         self.doc_word_sum = np.zeros(data_dict.docs_count,dtype="int")
#
#         # doc_word_topic 每篇文章每个词的类别 size: len(docs),len(doc)
#         # theta 文章->类的概率分布 size: len(docs), topic_num
#         # phi 类->词的概率分布 size: topic_num, len(doc)
#         self.doc_word_topic = \
#             np.array([[0 for y in range(data_dict.docs[x].length)] \
#                 for x in range(data_dict.docs_count)])
#         self.theta = np.array([[0.0 for y in range(self.topic_num)] \
#             for x in range(self.data_dict.docs_count)])
#         self.phi = np.array([[0.0 for y in range(self.data_dict.words_count)] \
#             for x in range(self.topic_num)])
#
#         #随机分配类型
#         for doc_idx in range(len(self.doc_word_topic)):
#             for word_idx in range(self.data_dict.docs[doc_idx].length):
#                 topic = random.randint(0,self.topic_num - 1)
#                 self.doc_word_topic[doc_idx][word_idx] = topic
#                 # 对应矩阵topic内容增加
#                 word = self.data_dict.docs[doc_idx].words[word_idx]
#                 self.word_topic_num[word][topic] += 1
#                 self.doc_topic_num[doc_idx][topic] += 1
#                 self.doc_word_sum[doc_idx] += 1
#                 self.topic_word_sum[topic] += 1
#
#     def sampling(self, doc_idx, word_idx):
#
#         topic = self.doc_word_topic[doc_idx][word_idx]
#         word = self.data_dict.docs[doc_idx].words[word_idx]
#         # Gibbs 采样，是去除上一次原本情况的采样
#         self.word_topic_num[word][topic] -= 1
#         self.doc_topic_num[doc_idx][topic] -= 1
#         self.topic_word_sum[topic] -= 1
#         self.doc_word_sum[doc_idx] -= 1
#         # 构造计算公式
#         Vbeta = self.data_dict.words_count * self.beta
#         Kalpha = self.topic_num * self.alpha
#         self.p = (self.word_topic_num[word] + self.beta) / \
#                     (self.topic_word_sum + Vbeta) * \
#                  (self.doc_topic_num[doc_idx] + self.alpha) / \
#                     (self.doc_word_sum[doc_idx] + Kalpha)
#
#         for k in range(1,self.topic_num):
#             self.p[k] += self.p[k-1]
#         # 选取满足本次抽样的topic
#         u = random.uniform(0,self.p[self.topic_num - 1])
#         for topic in range(self.topic_num):
#             if self.p[topic] > u:
#                 break
#         # 将新topic加回去
#         self.word_topic_num[word][topic] += 1
#         self.doc_topic_num[doc_idx][topic] += 1
#         self.topic_word_sum[topic] += 1
#         self.doc_word_sum[doc_idx] += 1
#
#         return topic
#
#     def _theta(self):
#         for i in range(self.data_dict.docs_count):
#             self.theta[i] = (self.doc_topic_num[i]+self.alpha)/ \
#             (self.doc_word_sum[i]+self.topic_num * self.alpha)
#     def _phi(self):
#         for i in range(self.topic_num):
#             self.phi[i] = (self.word_topic_num.T[i] + self.beta)/ \
#             (self.topic_word_sum[i]+self.data_dict.words_count * self.beta)
#
#     def train_lda(self):
#         for x in range(self.iter_times):
#             for i in range(self.data_dict.docs_count):
#                 for j in range(self.data_dict.docs[i].length):
#                     topic = self.sampling(i,j)
#                     self.doc_word_topic[i][j] = topic
#         print("迭代完成。")
#         print("计算文章-主题分布")
#         self._theta()
#         print("计算词-主题分布")
#         self._phi()
#
# def main():
#     data_clean = DataClean(train_file)
#     data_clean.clean()
#     data_dict = data_clean.data_dict
#     data_dict.save_word2id(bag_word_file)
#     lda = LDAModel(data_dict)
#     lda.train_lda()
#
# if __name__ == '__main__':
#     main()
#
