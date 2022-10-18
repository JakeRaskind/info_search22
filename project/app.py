import streamlit as st
from time import time
from construct_index import index


def app():
    st.write("""
    <style>app
    @import url('https://fonts.googleapis.com/css2?family=Pacifico');
    html, body, [class*="css"]  {
    font-family: 'Pacifico';
    }
    </style>
    """, unsafe_allow_html=True)

    st.title('Ответы mail.ru о :cupid: любви :cupid:')
    st.write('Искусственный интеллект в мягкой обложке')

    query = st.text_input('Чего ищем?')

    left, right = st.columns(2)
    ans_num = right.number_input(label='Сколько чудесных ответов хотим увидеть?', min_value=1, max_value=50, value=5, step=1)
    alg = left.radio('Как ищем?', ['bert', 'bm', 'tfidf'])
    if st.button('Search'):
        start = time()
        index.mode = alg
        ans_list = index.process_query(query)
        st.subheader('Вот что об этом думают компетентные эксперты:')
        container = st.container()
        for ans in ans_list[:ans_num]:
            container.write(ans)
        st.write(str(f'\nЭксперты ответили за {time() - start} секунд'))


if __name__ == '__main__':
    app()