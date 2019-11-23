from flask import Flask, Flask, redirect, url_for, request, jsonify
from gensim.models import Word2Vec
from konlpy.tag import Okt
import operator
import pickle
import w2v_similarity_corpus_s5 as embedding
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

data = data[1:]


@app.route('/test', methods=['POST'])
def hello():
    usersay = request.get_json(force=True)
    print(usersay)
    user_say = usersay["chatText"]
    result = embedding.embedding_distance(user_say, data, 5)

    return jsonify(str(result))


if __name__ == '__main__':
    app.run(debug=True)
