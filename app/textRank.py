from app.wordCut import KeyWord
import networkx as nx
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt

def caculDistance(a,c,type='cos'):
    '''
    计算本簇到簇c的中心距离
    :param c: 另外一个簇
    :param type: 距离类型，eu为欧氏距离，mh为曼哈顿距离，qb为切比雪夫，cos为余弦距离
    :return: 距离
    '''
    a=np.array(a)
    c=np.array(c)
    if type == 'eu':
        return np.linalg.norm(a - c)
    elif type == 'mh ':
        return np.linalg.norm(a - c, ord=1)
    elif type == 'qb':
        return np.linalg.norm(a - c, ord=np.inf)
    elif type == 'cos':
        return np.dot(a, c) / (np.linalg.norm(a) * (np.linalg.norm(c)))
    else:
        return None

model = KeyedVectors.load_word2vec_format('../data/W2VModel_self')
kw = KeyWord()
fout=open('res_100.txt','w',encoding='utf8')
for i in range(100):
    print(i)
    docid,title,kv=kw(i)
    fout.write('\n')
    fout.write('Article:{}\t{}\n'.format(docid,title))
    fout.write("KeyWordBefore:{}\n".format(' '.join([node+':'+str(v) for node,v in sorted(kv.items(),key = lambda x:x[1],reverse = True)])))
    data={}
    # with open("kvdict.txt",'r',encoding='utf8') as  f:
    #     model=eval(f.read().replace('array(','').replace(',\n      dtype=float32)',''))
    G=nx.Graph()
    i=0
    kv_d={}
    for k,v in kv.items():
        if k in model:
            G.add_node(k,weight=v)
            kv_d[k]=v
            for kk,vv in list(kv.items())[i+1:]:
                if kk in model:
                    similar=caculDistance(model[k],model[kk])
                    if similar>0.5 and similar<10:
                        G.add_edge(k,kk,weight=similar)
    # fout.write('\n'.join([str(t) for t in sorted(G.edges.data(),key=lambda x: x[0], reverse=True)]))
    pr=nx.pagerank(G,nstart=kv_d)
    node_dict=G.nodes(data='weight')
    fout.write("KeyWordAfter:{}\n".format(' '.join([node+':'+str(v) for node,v in sorted(pr.items(),key = lambda x:x[1],reverse = True)])))
    # nx.draw(G, with_labels=True, label_size=500, node_size=500, font_size=10)
    i=1
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        if len(c)>3:
            fout.write("Cluster{}:{}\n".format(str(i),
                ' '.join([node + ':' + str(v) for node, v in sorted([(item,node_dict[item]) for item in c],key=lambda x: x[1], reverse=True)])))
            i+=1
        else:
            break
    # plt.show()
