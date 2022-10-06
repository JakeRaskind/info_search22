from preprocessing import preprocess_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy import sparse


class MatrixIndex():
    def __init__(self, texts):

        def _bm_count(tfs, idfs):
            doc_lens = tfs.sum(axis=1)
            avg = doc_lens.mean()
            mapping = sparse.find(tfs)
            k = 2
            b = 0.75
            bms = []
            for i in range(len(tfs.data)):
                bms.append(idfs[mapping[0][i]]*(mapping[2][i] * (k+1))/(mapping[2][i] + k*(1-b+b*doc_lens[mapping[0][i]]/avg)))
            return sparse.csr_matrix((np.array(bms).ravel(), (mapping[0].ravel(), mapping[1].ravel())))

        self.texts = np.array(texts)
        tfidf_vect = TfidfVectorizer(analyzer='word', use_idf=True)
        self.tf_vect = CountVectorizer(analyzer='word')
        tfidf_vect.fit(texts)
        tfs = self.tf_vect.fit_transform((texts))
        self.index = _bm_count(tfs, tfidf_vect.idf_)



    def parse_query(self, query) -> int:
        '''
        возвращает суммарное количество употреблений слов из запроса в индексе
        :param query:
        :return:
        '''
        tokens = preprocess_file(query)
        return self.tf_vect.transform([query])

    @staticmethod
    def bm_similarity(matrix , vec) -> np.ndarray:
        '''
        считает bm близость вектора и каждой строчки индекса
        :param matrix: numpy.ndarray
        :param vec: numpy.ndarray
        :return:
        '''
        return matrix @ vec.reshape(-1, 1)

    def process_query(self, query) -> list:
        '''
        Возвращает список answers в порядке похожести на запрос
        :param query:
        :return:
        '''
        q_vec = self.parse_query(query)
        dists = self.bm_similarity(self.index, q_vec).toarray()
        return self.texts[np.argsort(dists, axis=0)[::-1]].ravel().tolist()