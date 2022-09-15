import os


def read_files(directory):
    '''
    чтение файлов из директории
    :return:
    '''
    for root, dirs, files in os.walk(directory):
        for name in files:
            filepath = os.path.join(root, name)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                yield name, text