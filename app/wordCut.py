import jieba
import pandas as pd

def combinedata():
    part1=pd.read_csv('../data/input/data-灾害-1000000.csv')
    part2=pd.read_csv('../data/input/data-灾害-1000000-2.csv')
    part3=pd.read_csv('../data/input/灾害-1000.csv')
    data=pd.concat([part2,part1,part3])#合并数据
    assert part1.shape[0]+part2.shape[0]+part3.shape[0]== data.shape[0]
    data=pd.DataFrame(data)
    print(data.shape)
    data=data.drop_duplicates('title')#根据标题去重
    print(data.shape)
    data.to_csv('../data/input/data-all.csv',index=False)
def cutit():
    data=pd.read_csv('../data/input/data-all.csv')
    data=data[['title','content']]
    jieba.load_userdict('../data/pre.dict') #加载自定义词典
    print('load dict finished!')
    with open('../data/stopWords.txt','r',encoding='utf8') as f:
        stopwords=[x.strip() for x in f.readlines()]
    data['content']=data.apply(lambda r:' '.join(jieba.cut(r['content'].replace(' ','').replace('\t','').replace('\n','').replace(',','，'))),axis=1)
    print('data cut finished!')
    data['title']=data.apply(lambda t:t['title'].replace(' ','').replace(',','，'),axis=1)
    data.to_csv('../data/output/data-finally.csv',header=False)
def w2v():
    pass

if __name__=='__main__':
    # combinedata()
    cutit()
