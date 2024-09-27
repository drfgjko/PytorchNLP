import json
import os
import numpy as np
from gensim.models.word2vec import Word2Vec
from opencc import OpenCC
import pickle


# json->txt
def json_to_txt():
    converter = OpenCC('t2s')
    txt = './txt_dataset/song.txt'

    with open(txt, 'w') as f:
        f.write('')
    for i in range(100):
        path = './origin_dataset/poet.song.' + str(i * 1000 + 7000) + '.json'
        # 读取json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 追加写入txt
        with open(txt, 'a', encoding='utf-8') as f:
            for item in data:
                # 获取"paragraphs"
                paragraphs = ''.join(map(str, item.get('paragraphs', '')))
                # 单句有5字或7字，保留统一格式的诗->七言绝句
                sentences = [sentence.strip() for sentence in paragraphs.split('。') if sentence.strip()]
                flag = 1
                # 只处理长度为2的诗
                if len(sentences) == 2:
                    for sentence in sentences:
                        # 单句有16字，保留统一格式的诗
                        if len(sentence) != 15:
                            flag = 0
                            break
                    if flag == 1:
                        # paragraphs从繁体->简体，并写入到 test.txt 文件中
                        paragraphs_simplified = converter.convert(paragraphs.strip())
                        f.write(paragraphs_simplified + '\n')


# 拆分(每个字以空格分开) -> 存入'./dataset/song_split.txt'
def split_poetry(file='./txt_dataset/song.txt'):
    txt_split_file = './dataset/song_split.txt'
    data = open(file, "r", encoding="utf-8").read()
    # 以空格拆分每个字
    all_date_split = " ".join(data).strip()
    with open(txt_split_file, 'w', encoding="utf-8") as f:
        f.write(all_date_split)


# 返回w1只是为拿shape
# 词向量,获取: data,w1,key_to_index,index_to_key
def train_vec(split_file='./dataset/song_split.txt', org_file='./txt_dataset/song.txt'):
    vec_params_file = 'vec_params.pkl'

    if not os.path.exists(org_file):
        json_to_txt()
    if not os.path.exists(split_file):
        split_poetry()
    split_data = open(split_file, 'r', encoding="utf-8").read().split("\n")
    # 还没进行创建词汇表等工作
    if not os.path.exists(vec_params_file):
        # vector_size = embedding_dim -> 词向量的维度
        model = Word2Vec(vector_size=110, min_count=1, sg=1, hs=0, workers=10)
        # 构建词汇表
        model.build_vocab(split_data)
        # model.syn1neg = w1
        # (vocab_size, embedding_dim)

        # model.wv.key_to_index: 这是一个字典，将词汇表中的单词映射到其对应的索引。
        # model.wv.index_to_key: 这是一个列表 dict(enumerate(model.wv.index_to_key)-> 变成跟model.wv.key_to_index形式类似的字典
        # model.wv.index_to_key:模型中的所有单词列表:单词列表是唯一的->每个单词只会出现一次

        pickle.dump((model.syn1neg, model.wv.key_to_index, dict(enumerate(model.wv.index_to_key))), open("vec_params"
                                                                                                         ".pkl",
                                                                                                         "wb"))
        # 修改成后续训练时需要的数据格式
        # poem_array:一个二维列表
        # shape:(num_poems, max_poem_length)
        poem_indices = [[model.wv.key_to_index[word] for word in poem.split()] for poem in split_data]
        poem_array = np.array(poem_indices)

        return poem_array, (model.syn1neg, model.wv.key_to_index, dict(enumerate(model.wv.index_to_key)))

    syn1neg, key_to_index, index_to_key = pickle.load(open(vec_params_file, 'rb'))

    poem_indices = [[key_to_index[word] for word in poem.split()] for poem in split_data]
    poem_array = np.array(poem_indices)

    # vec_params_file 里存的是(model.syn1neg, model.wv.key_to_index, model.wv.index_to_key)
    return poem_array, (syn1neg, key_to_index, index_to_key)
