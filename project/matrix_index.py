from preprocessing import preprocess_file
from file_reader import read_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy import sparse
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import os
from zipfile import ZipFile


class IndexDataset(Dataset):
    def __init__(self, texts):
        self.data = texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MatrixIndex():
    def __init__(self, IND_PATH='', texts=[], mode='bert'):

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


        self.tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
        self.model = AutoModel.from_pretrained('sberbank-ai/sbert_large_nlu_ru')
        self.mode = mode

        if os.path.exists(IND_PATH):
            self.texts = np.array(read_file(os.path.join(IND_PATH, 'mail_love.jsonl')))
            with ZipFile(os.path.join(IND_PATH, 'tfidf_index.zip')) as myzip:
                with myzip.open('content/tfidf_index.pkl', 'r') as f:
                    self.tfidf_index = pickle.load(f)
            with ZipFile(os.path.join(IND_PATH, 'bert_index.zip')) as myzip:
                with myzip.open('content/bert_index.pkl', 'r') as f:
                    self.bert_index = pickle.load(f)
            with ZipFile(os.path.join(IND_PATH, 'bm_index.zip')) as myzip:
                with myzip.open('content/bm_index.pkl', 'r') as f:
                    self.bm_index = pickle.load(f)
            with open(os.path.join(IND_PATH, 'tf_vect.pkl'), 'rb') as f:
                self.tf_vect = pickle.load(f)
            with open(os.path.join(IND_PATH, 'tfidf_vect.pkl'), 'rb') as f:
                self.tfidf_vect = pickle.load(f)
        else:
            self.texts = np.array(texts)
            self.device = 'cuda:0'
            self.tfidf_vect = TfidfVectorizer(analyzer='word', use_idf=True)
            self.tf_vect = CountVectorizer(analyzer='word')
            self.tfidf_index = self.tfidf_vect.fit_transform(texts)
            tfs = self.tf_vect.fit_transform(texts)
            self.bm_index = _bm_count(tfs, self.tfidf_vect.idf_)
            self.bert_index = self.bert_transform_index(self.texts)

    def bert_transform_index(self, texts):

        def _collate_fn(batch):
            return tokenizer.batch_encode_plus(batch, return_tensors='pt', **encode_plus_kwargs)

        dataset = IndexDataset(texts)
        res = np.zeros((len(dataset), self.model.pooler.dense.out_features))
        encode_plus_kwargs = {'max_length': 24, 'truncation': True, 'padding': True}
        dataloader = DataLoader(dataset, batch_size=128, collate_fn=_collate_fn)
        tokenizer = self.tokenizer
        self.model.to(self.device)
        self.model.eval()
        for i, batch in enumerate(tqdm(dataloader)):
            batch.to(self.device)
            outputs = self.model(**batch)
            sent_embs = self.mean_pooling(outputs, batch['attention_mask']).detach().cpu().numpy()
            res[i:i+batch['input_ids'].shape[0],:] = sent_embs
        return res

    def bert_transform_query(self, query):
      encode_plus_kwargs = {'max_length': 24, 'truncation': True, 'padding': True}
      encoded_query = self.tokenizer.batch_encode_plus(query, return_tensors='pt', **encode_plus_kwargs)
      self.model.to('cpu')
      with torch.no_grad():
        outputs = self.model(**encoded_query)
      sent_embs = self.mean_pooling(outputs, encoded_query['attention_mask']).detach().numpy()
      return sent_embs

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def parse_query(self, queries) -> int:
        '''
        возвращает суммарное количество употреблений слов из запроса в индексе
        :param query:
        :return:
        '''
        if self.mode == 'bert':
            return self.bert_transform_query(queries)
        else:
          tokens = np.array([preprocess_file(query) for query in queries])
          if self.mode == 'bm':
            return self.tf_vect.transform(tokens)
          elif self.mode == 'tfidf':
            return self.tfidf_vect.transform(tokens)


    @staticmethod
    def similarity(matrix , vec) -> np.ndarray:
        '''
        считает bm близость вектора и каждой строчки индекса
        :param matrix: numpy.ndarray or sparse
        :param vec: numpy.ndarray or sparse
        :return:
        '''
        return matrix @ vec.T

    def process_query(self, query) -> list:
        '''
        Возвращает список answers в порядке похожести на запрос
        :param query:
        :return:
        '''
        if isinstance(query, str):
          query = [query]
        q_vec = self.parse_query(query)
        if self.mode == 'bert':
          dists = self.similarity(self.bert_index, q_vec)
        elif self.mode == 'bm':
          dists = self.similarity(self.bm_index, q_vec).toarray()
        elif self.mode == 'tfidf':
          dists = self.similarity(self.tfidf_index, q_vec).toarray()
        return self.texts[np.argsort(dists, axis=0)[::-1]].ravel().tolist()

    def topNscore(self, n, queries):
        samp_num = len(queries)
        if self.mode == 'bert':
            parsed_queries = self.bert_transform_index(queries)
        if self.mode == 'bm':
            parsed_queries = self.parse_query(queries)
        overall_scores = self.similarity(self.index, parsed_queries)
        if self.mode == 'bm':
          overall_scores = overall_scores.toarray()
        top_n = np.argsort(overall_scores, axis=0)[::-1][:n, :]
        matches = np.any(top_n == np.arange(samp_num), axis=0)
        return sum(matches) / samp_num
