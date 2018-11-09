# coding: utf-8 
import re
from mrjob.job import MRJob
import math
from mrjob.step import MRStep
class CalCU(MRJob):
    def mapper1(self, _, line):
        docid,title,content=re.split(',',line,2)
        ws=content.strip().split(' ')
        for x in set(ws):
            yield x,(docid,(ws.count(x),ws.index(x),ws[::-1].index(x),1 if title.find(x)==-1 else 1.5,len(ws)))
    def reducer1(self, key, value):
        for v in value:
            yield key, v
    def reducer2(self, key, values):
        N=10000
        values=list(values)
        M=len(values)
        for k,v in values:
            if len(str(key))>0:
                res=float(math.log(len(str(key)),2)*(float(1)+math.exp(-float(v[1])))*(float(v[0])/float(v[-1]))*math.log(float(N)/float(M))*(1+(float(v[-1])-float(v[2])-float(v[1]))/float(v[-1]))*float(v[-2]))
                yield str(k),(str(key),res*100)
    def reducer3(self, key, value):
        value=list(value)
        print('{}\t{}'.format(key,' '.join([x[0]+':'+str(x[1]) for x in sorted(value,key=lambda x:x[1],reverse = True)])))
        #yield key,sorted(value,key=lambda x:x[1],reverse = True)
    def steps(self):
        #return [MRStep(mapper=self.mapper1, combiner=self.reducer1, reducer=self.reducer1),MRStep(reducer=self.reducer2)]
        return [MRStep(mapper=self.mapper1, combiner=self.reducer1, reducer=self.reducer1),MRStep(reducer=self.reducer2),MRStep(reducer=self.reducer3)]
if __name__=='__main__':
    CalCU.run()
