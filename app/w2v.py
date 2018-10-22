from gensim.models import KeyedVectors
import pickle,numpy

# model = KeyedVectors.load_word2vec_format('../data/W2VModel_self')
# with open('../data/w2v_pkl','wb') as f:
#     pickle.dump(model,f)
with open('../data/w2v_pkl','rb') as f:
    model=pickle.load(f)
for w,v in model.similar_by_word('海啸',10):
    print('{}:{}'.format(w,str(v)))
# print(numpy.array(model['我们']).shape)