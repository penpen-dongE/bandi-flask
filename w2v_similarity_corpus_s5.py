from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import operator
import pickle

okt = Okt()
model = Word2Vec.load('w2v_s5.model')

w2v_vocab = model.wv.vocab
w2v_vocab = list(w2v_vocab.keys())


def nlp_sentence(sentence):
    tokens = okt.pos(sentence, norm=True, stem=True)
    result = [word[0]
              for word in tokens if word[1] != 'Josa' and word[0] in w2v_vocab]
    return result


def embedding_distance(user_say, data, num=5, policy=0):
    result = []
    s1 = nlp_sentence(user_say)
    for i in data:
        s2 = i[6]
        if s2:
            s = model.wv.n_similarity(s1, s2)
        result.append((s, i[1]))
    result = sorted(result, key=operator.itemgetter(0), reverse=True)
    return result[policy:num+policy]


with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

data = data[1:]
