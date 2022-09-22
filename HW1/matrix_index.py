from preprocessing import preprocess_file
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class MatrixIndex():
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(analyzer='word')
        self.index = self.vectorizer.fit_transform(list(map(lambda x: ' '.join(x), texts.values()))).toarray()
        self.id2names = np.array(list(texts.keys()))

    def parse_query(self, query) -> int:
        '''
        возвращает суммарное количество употреблений слов из запроса в индексе
        :param query:
        :return:
        '''
        tokens = preprocess_file(query)
        return self.vectorizer.transform([' '.join(tokens)]).toarray().ravel()

    @staticmethod
    def cosine_similarity(matrix , vec) -> np.ndarray:
        '''
        считает косинусную близость вектора и каждой строчки индекса
        :param matrix: numpy.ndarray
        :param vec: numpy.ndarray
        :return:
        '''
        return np.dot(matrix, vec.reshape(-1, 1)).ravel()

    def process_query(self, query) -> list:
        '''
        Возвращает список серий в порядке похожести на запрос
        :param query:
        :return:
        '''
        q_vec = self.parse_query(query)
        dists = self.cosine_similarity(self.index, q_vec)
        return self.id2names[-np.argsort(-dists)].tolist()