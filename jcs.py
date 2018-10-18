from math import log
import operator
'''创建数据'''
def creatDataSet():
    dataSet=[
            [1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']
    ]
    labels=['no surfacing','flippers']
    #返回数据集和类标签
    return dataSet,labels
'''计算熵'''
def calEnt(dataSet):
    #获取数据的行数
    lenDataSet = len(dataSet)
    p = {}
    #初始化熵
    H = 0.0
    for data in dataSet:
        curLabel = data[-1]
        if curLabel not in p.keys():
            p[curLabel] = 0
        p[curLabel] +=1
    for key in p:
        px = float(p[key])/float(lenDataSet)
        #计算各个熵
        H -= px *log(px,2)   
    return  H
'''划分数据集'''
def splitDataSet(dataSet,axis,value):
    retDataSet = [] 
    for data in dataSet:
        if data[axis] == value:
            subData = data[:axis]
            subData.extend(data[axis+1:])
            retDataSet.append(subData)
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    #特征数量
    numFeature = len(dataSet[0]) - 1
    #计算基础熵
    baseEnt = calEnt(dataSet)
    bestEnt = 0.0
    bestFeature = -1
    for i in range(numFeature):
        #得到第i位上所有的特征值
        featList=[example[i] for example in dataSet]
        newEnt = 0.0
        for feature in set(featList):
            data = splitDataSet(dataSet, i, feature)
            prob = float(len(data))/float(len(dataSet))
            newEnt += prob*calEnt(data)
        infoEnt = baseEnt - newEnt
        if infoEnt > bestEnt:
            bestEnt = infoEnt
            bestFeature = i
    
    return bestFeature
'''多数表决'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items)
    return sortedClassCount[0][0]        
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #所有的都属于同一类
    if classList.count(classList[0]) == len(dataSet):
        return dataSet[0]
    #遍历完所有的特征集 此时特征只有一个时 则采用表决的结果
    if len(dataSet[0]) == 1:
        return majorityCnt[dataSet]
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree={bestFeatLabel:{}}
    subLabels=labels[:]
    del(subLabels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        #@采用递归的方法利用该特征对数据集进行分类
        #@bestFeatLabel 分类特征的特征标签值
        #@dataSet 要分类的数据集
        #@bestFeat 分类特征的标称值
        #@value 标称型特征的取值
        #@subLabels 去除分类特征后的子特征标签列表
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
#------------------------测试算法------------------------------    
#完成决策树的构造后，采用决策树实现具体应用
#@intputTree 构建好的决策树
#@featLabels 特征标签列表
#@testVec 测试实例
def classify(inputTree,featLabels,testVec):
    #找到树的第一个分类特征，或者说根节点'no surfacing'
    #注意python2.x和3.x区别，2.x可写成firstStr=inputTree.keys()[0]
    #而不支持3.x
    firstStr=list(inputTree.keys())[0]
    #从树中得到该分类特征的分支，有0和1
    secondDict=inputTree[firstStr]
    #根据分类特征的索引找到对应的标称型数据值
    #'no surfacing'对应的索引为0
    featIndex=featLabels.index(firstStr)
    #遍历分类特征所有的取值
    for key in secondDict.keys():
        #测试实例的第0个特征取值等于第key个子节点
        if testVec[featIndex]==key:
            #type()函数判断该子节点是否为字典类型
            if type(secondDict[key]).__name__=='dict':
                #子节点为字典类型，则从该分支树开始继续遍历分类
                classLabel=classify(secondDict[key],featLabels,testVec)
            #如果是叶子节点，则返回节点取值
            else: classLabel=secondDict[key]
    return classLabel

#决策树的存储：python的pickle模块序列化决策树对象，使决策树保存在磁盘中
#在需要时读取即可，数据集很大时，可以节省构造树的时间
#pickle模块存储决策树
def storeTree(inputTree,filename):
    #导入pickle模块
    import pickle
    #创建一个可以'写'的文本文件
    #这里，如果按树中写的'w',将会报错write() argument must be str,not bytes
    #所以这里改为二进制写入'wb'
    fw=open(filename,'wb')
    #pickle的dump函数将决策树写入文件中
    pickle.dump(inputTree,fw)
    #写完成后关闭文件
    fw.close()
#取决策树操作    
def grabTree(filename):
    import pickle
    #对应于二进制方式写入数据，'rb'采用二进制形式读出数据
    fr=open(filename,'rb')
    return pickle.load(fr)
#------------------------示例：使用决策树预测隐形眼镜类型----------------
def predictLensesType(filename):
    #打开文本数据
    fr=open(filename)
    #将文本数据的每一个数据行按照tab键分割，并依次存入lenses
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    #创建并存入特征标签列表
    lensesLabels=['age','prescript','astigmatic','tearRate']
    #根据继续文件得到的数据集和特征标签列表创建决策树
    lensesTree=createTree(lenses,lensesLabels)
    return lensesTree

dataSet,labels = creatDataSet()
# print(splitDataSet(dataSet, 0, 1))
# print(calEnt(dataSet))
myTree  = createTree(dataSet,labels)
print(classify(myTree, labels, [1,0]))
print(classify(myTree, labels, [1,1]))
print(classify(myTree, labels, [0,1]))
# print(majorityCnt([1,1,1,1,2,2]))
# {
#     'no surfacing': {
#         0: [1, 'no'], 
#         1: {
#             'flippers':
#                 {
#                     0: ['no'], 
#                     1: ['yes']
#                 }
#            }
#     }
# }
