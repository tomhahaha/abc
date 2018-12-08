import numpy as np
from gensim.models import KeyedVectors
import os
model = KeyedVectors.load_word2vec_format('../data/W2VModel_self')
class Clust():
    def __init__(self,nodes):
        '''
        :param nodes: [(word,val,vec)...]
        '''
        self.items=nodes
        self.length=len(nodes)
        self.center=self.getClustCenter()
        self.articlenum=1
    def getClustCenter(self):
        return np.mean([np.array(y)*x for _,x,y in self.items],axis=0)
    def addNodes(self,nodes):
        self.items+=nodes
        self.center=(self.center*self.length+sum([np.array(node[2])*node[1] for node in nodes]))/(self.length+len(nodes))
        self.length+=len(nodes)
    def popNodes(self,indexs):
        res=[self.items.pop(i) for i in indexs]
        self.center=self.getClustCenter()
        self.length=len(self.items)
        return res
    def contact(self,c):
        '''合并本簇与c'''
        self.center=(self.center*self.length+c.center*c.length)/(self.length+c.length)
        self.items=self.items+c.items
        self.length=self.length+c.length
        self.articlenum=self.articlenum+c.articlenum
        del c
        return self
    def showItem(self):
        one={}
        for x, y, _ in self.items:
            if x in one:
                one[x]+=y
            else:
                one[x]=y
        return ' '.join(['{}:{}'.format(x,str(y)) for x,y in one.items()])
    def caculDistance(self,c,type='eu'):
        '''
        计算本簇到簇c的中心距离
        :param c: 另外一个簇
        :param type: 距离类型，eu为欧氏距离，mh为曼哈顿距离，qb为切比雪夫，cos为余弦距离
        :return: 距离
        '''
        if type=='eu':
            return np.linalg.norm(self.center-c.center)
        elif type=='mh ':
            return np.linalg.norm(self.center-c.center,ord=1)
        elif type=='qb':
            return np.linalg.norm(self.center-c.center,ord=np.inf)
        elif type=='cos':
            return 1-np.dot(self.center,c.center) / (np.linalg.norm(self.center) * (np.linalg.norm(c.center)))
        else:
            return None
def extraData(f):
    with open(f,'r',encoding='utf8') as fin:
        data=fin.readlines()
    return [Clust([(x.split('/n:')[0],float(x.split('/n:')[1]),model[x.split('/n:')[0]]) for x in y.strip().split('\t')[1].split(' ') if x.split('/n:')[0] in model]) for y in data]
def train_once(clustlist,m):
    i=0
    while i<len(clustlist)-1:
        j=i+1
        while j<len(clustlist):
            if clustlist[i].caculDistance(clustlist[j],'cos')<m:
                clustlist[i]=clustlist[i].contact(clustlist.pop(j))
            else:
                j+=1
        i+=1
    return clustlist
def train(clusterlist,m):
    print('------------begin train！-------------')
    lenth=len(clusterlist)
    epoch=1
    while 1:
        print('第{}次训练，簇数量：{}'.format(str(epoch),str(lenth)))
        clusterlist=train_once(clusterlist,m)
        if lenth==len(clusterlist):
            break
        else:
            lenth=len(clusterlist)
            epoch+=1
    print('------------train over！-------------')
    return clusterlist

if __name__=='__main__':
    n=0.1
    while n!=0.0:
        cl = []
        print('------------load data！-------------')
        for root, dir, fs in os.walk('../data/doc_split'):
            for f in fs:
                print(f)
                cl += extraData('../data/doc_split/' + f)
        res=train(cl,n)
        fout=open('../clust_res.txt','w',encoding='utf8')
        for i,cl in enumerate(res):
            fout.write("{}:\t{}\t{}\n".format(str(i),str(cl.articlenum),cl.showItem()))
        fout.close()
        n=float(input())