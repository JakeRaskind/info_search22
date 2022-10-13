from matrix_index import MatrixIndex
from preprocessing import preprocess_file
from file_reader import read_file, read_questions
import numpy as np


def top5test():
    corpus = read_file(args.directory)
    bert_index = MatrixIndex(corpus, mode='bert')
    bm_index = MatrixIndex(corpus, mode='bm')
    quests = np.array(read_questions(args.directory))
    bert_top5 = bert_index.topNscore(5, quests)
    bm_top5 = bm_index.topNscore(5, quests)
    print(f'Bert top5score: {bert_top5}\nBM25 top5score: {bm_top5}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Enter data directory')
    args = parser.parse_args()
    main()
