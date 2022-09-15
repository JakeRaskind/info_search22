from preprocessing import *


class DictIndex():
    def __init__(self, texts):
        self.index = {}
        self.corp_size = len(texts)
        for name, text in texts.items():
            for word in text:
                self.index[word] = self.index.get(word, {})
                self.index[word][name] = self.index[word].get(name, 0) + 1
        self.counts = {k: sum(v.values()) for (k, v) in self.index.items()}


    def most_frequent(self) -> str:
        '''
        возвращает самое частотное слово
        :return:
        '''
        return max(self.counts.keys(), key=lambda x: self.counts[x])

    def least_frequent(self) -> str:
        '''
        возвращает одно из наменее частотных слов
        :return:
        '''
        return min(self.counts.keys(), key=lambda x: self.counts[x])

    def process_query(self, query) -> int:
        '''
        возвращает суммарное количество употреблений слов из запроса в индексе
        :param query:
        :return:
        '''
        tokens = preprocess_file(query)
        return sum([self.counts.get(word, 0) for word in tokens])

    def universal_tokens(self) -> list:
        '''
        возвращает слова, встреченные во всех докуметах индекса
        :return:
        '''
        return [word for word in self.index.keys() if len(self.index[word]) == self.corp_size]
