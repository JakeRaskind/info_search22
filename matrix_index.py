from preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class MatrixIndex():
    def __init__(self, texts):
        self.vectorizer = CountVectorizer(analyzer='word')
        self.index = self.vectorizer.fit_transform(list(map(lambda x: ' '.join(x), texts.values())))

    def most_frequent(self) -> str:
        '''
        возвращает самое частотное слово
        :return:
        '''
        return self.vectorizer.get_feature_names_out()[np.argmax(self.index.sum(axis=0))]

    def least_frequent(self) -> str:
        '''
        возвращает одно из наменее частотных слов
        :return:
        '''
        return self.vectorizer.get_feature_names_out()[np.argmin(self.index.sum(axis=0))]

    def process_query(self, query) -> int:
        '''
        возвращает суммарное количество употреблений слов из запроса в индексе
        :param query:
        :return:
        '''
        tokens = preprocess_file(query)
        return sum(self.index.toarray() @ self.vectorizer.transform([' '.join(tokens)]).toarray().reshape(-1, 1))

    def universal_tokens(self) -> list:
        '''
        возвращает слова, встреченные во всех докуметах индекса
        :return:
        '''
        return self.vectorizer.get_feature_names_out()[[np.nonzero(np.prod(self.index.toarray() != 0, axis=0))]].ravel()