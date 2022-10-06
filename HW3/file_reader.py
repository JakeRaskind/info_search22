import os
import json

def read_file(directory):
    '''
    чтение файлов из директории
    :return:
    '''
    with open(directory, 'r', encoding='utf-8') as f:
        data = list(f)[:50000]
    corpus = [max(json.loads(obj)['answers'], key=lambda x: x['author_rating']['value'])['text'] for obj in data if json.loads(obj)['answers']]
    return corpus
