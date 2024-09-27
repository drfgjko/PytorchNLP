import pickle
import random
import torch
from opencc import OpenCC
from torch.autograd import Variable
from data import train_vec
from mymodel_lstm import MyLstmModel
from mymodel_rnn import MyRnnModel
from train import get_args, train
import torch.nn.functional as f

device = "cuda" if torch.cuda.is_available() else "cpu"

vec_params_file = 'vec_params.pkl'
org_file = './txt_dataset/song.txt'

# 默认是Test
# 1: 训练lstm 2：训练rnn 3：test
mode = 3


# mode = 3 -> test
def test(start_word, model_type=1):
    if model_type == 2:
        # 加载模型并设置为eval
        model = load_model('./model/model_rnn')
        generated_poem = generate_poem(model, start_word)
        return generated_poem
    # 默认 使用LSTM
    model = load_model('./model/model_lstm')
    generated_poem = generate_poem(model, start_word)
    return generated_poem


# 加载模型
def load_model(file):
    all_data, (w1, word_2_index, index_2_word) = train_vec()
    _, hidden_dim, num_layers, _, _ = get_args()
    vocab_size, embedding_dim = w1.shape

    if file == './model/model_lstm':
        # MyLstmModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        model = MyLstmModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        # 实例化模型时
        model.load_state_dict(torch.load(file))
        # 设置为评估模式
        model.eval()
        return model

    elif file == './model/model_rnn':
        # MyRnnModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        model = MyRnnModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        # 实例化模型时
        model.load_state_dict(torch.load(file))
        # 设置为评估模式
        model.eval()
        return model


# 输入单字，生成的诗句格式比较规定
def generate_poem(model, input_char, max_length=31):
    (w1, word_2_index, index_2_word) = pickle.load(open(vec_params_file, 'rb'))

    result = []
    x_input = Variable(
        torch.Tensor([word_2_index[input_char]]).view(1, 1).long())

    x_input.to(device)
    model.to(device)

    h_0 = None
    c_0 = None
    word_index = word_2_index[input_char]

    if word_index == -1:
        return "请换个字输入！"

    result.append(input_char)

    if isinstance(model, MyRnnModel):
        for i in range(max_length):
            pre, h_0 = model(x_input, h_0)
            # word_index = int(torch.argmax(pre))
            # top_index = pre.data[0].topk(1)[1][0]
            top_7_probs, top_7_indices = torch.topk(pre[0], 7)

            probs = f.softmax(pre, dim=1)[0]
            top_7_probs = probs[top_7_indices]
            top_index = random.choices(top_7_indices, weights=top_7_probs, k=1)[0]

            pre_word = index_2_word[top_index.item()]

            result.append(pre_word)
            x_input = Variable(x_input.data.new([top_index])).view(1, 1)
        if len(result) < max_length:
            return "请换个字输入！"
        return ''.join(result)

    # 默认使用LSTM
    for i in range(max_length):
        pre, (h_0, c_0) = model(x_input, h_0, c_0)
        # word_index = int(torch.argmax(pre))
        # top_index = pre.data[0].topk(1)[1][0]
        top_7_probs, top_7_indices = torch.topk(pre[0], 7)

        probs = f.softmax(pre, dim=1)[0]
        top_7_probs = probs[top_7_indices]

        top_index = random.choices(top_7_indices, weights=top_7_probs, k=1)[0]
        pre_word = index_2_word[top_index.item()]

        result.append(pre_word)
        x_input = Variable(x_input.data.new([top_index])).view(1, 1)
    if len(result) < max_length:
        return "请换个字输入！"
    return ''.join(result)


def is_chinese(words):
    return '\u4e00' <= words <= '\u9fff'


if __name__ == '__main__':
    # 训练lstm
    # train(1)

    # 训练rnn
    # train(2)

    # train(mode)

    # _____________________________
    # print(all_data)
    # all_data: numpy数组 (poem_num,poem_words_num )
    # (诗歌的数量，每首诗歌的字数) -> (_,32)
    # print(np.shape(all_data))

    # all_data, (w1, word_2_index, index_2_word) = train_vec()
    # print(all_data)
    # print(w1)
    # print(word_2_index)
    # print(index_2_word)

    # word_2_index:dict  eg: '字':1
    # print(word_2_index)
    # index_2_word: 转成跟word_2_index相似的字典
    # print(index_2_word)

    # ________________________________
    converter = OpenCC('t2s')
    if mode == 3:
        while True:
            print("请输入一个字:", end='')
            word = input().strip()
            if not word:
                print("输入为空")
                break
            word = converter.convert(word[0])
            if not is_chinese(word):
                print("请输入中文")
                continue
            else:
                # test 默认使用LSTM，设为2则使用RNN
                # out_poem = test(word,2)
                out_poem = test(word, 2)
                print(out_poem)

    # ________________________________
    # score()
