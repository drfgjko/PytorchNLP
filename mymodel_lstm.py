import torch
import torch.nn as nn
from torch.autograd import Variable


# 定义神经网络模型
# 模型构建一个单向的LSTM层+一个Linear层
class MyLstmModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(MyLstmModel, self).__init__()

        self.hidden_dim = hidden_dim
        # 词嵌入层 设置权重矩阵大小
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.loss = nn.CrossEntropyLoss()
        # self.loss = torch.nn.NLLLoss()

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            batch_first=False, num_layers=self.num_layers,
                            dropout=0.4, bidirectional=False)

        # linear层:(hidden_dim,vocab_size) -> 映射进行维度转换得到目标(word_size*batch_size,vocab_size)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    # 前向传播逻辑
    def forward(self, data_input, h_0=None, c_0=None):
        seq_len, batch_size = data_input.size()
        # h_0 :tensor (num_Layers * num_directions,batch,hidden_num)
        # c_0 :tensor (num_Layers * num_directions,batch,hidden_num)
        if h_0 is None or c_0 is None:
            # 初始化 全0张量
            h_0 = data_input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = data_input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()

            # tensor->variable
            h_0, c_0 = Variable(h_0), Variable(c_0)

        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)

        # data_input:(seq_len,batch_size)
        # batch_first= False -> 输入的格式要求(seq_len, batch_size, feature)
        xs_embedding = self.embeddings(data_input)
        # xs_embedding.shape: (seq_len, batch_size, embedding_dim)

        pre, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))
        # hidden_drop = self.dropout(hidden)
        # flatten_hidden = self.flatten(hidden_drop)
        # pre = self.linear(flatten_hidden)

        # ((seq_len * batch_size),embedding_dim)
        pre = self.linear(pre.view(seq_len * batch_size, -1))
        return pre, (h_0, c_0)
