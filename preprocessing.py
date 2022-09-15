import nltk
nltk.download('stopwords')


from pymorphy2 import MorphAnalyzer
import re
from nltk.corpus import stopwords


morph = MorphAnalyzer()
sws = set(stopwords.words('russian')) | set(['-'])


def preprocess_file(text: str) -> list:
  '''
  приведение регистра и удаление лишнего, сплит текста, лемматизация.
  :param filename:
  :return:
  '''
  tokens = re.sub('[^а-яё\s-]', ' ', text.lower()).split()
  lemmas = [morph.parse(word)[0].normal_form for word in tokens if word not in sws]
  return lemmas