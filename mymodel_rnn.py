import torch
import torch.nn as nn
from torch.autograd import Variable


# 定义神经网络模型
# 模型构建一个RNN层+一个Linear层
class MyRnnModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(MyRnnModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.loss = nn.CrossEntropyLoss()
        # self.loss = torch.nn.NLLLoss()

        self.lstm = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim,
                           batch_first=False, num_layers=self.num_layers,
                           dropout=0.4, bidirectional=False)

        # linear层:[hidden_dim,vocab_size]
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, data_input, h_0=None):
        seq_len, batch_size = data_input.size()
        # h_0 :tensor (num_Layers * num_directions,batch,hidden_num)
        if h_0 is None:
            # h_0 = torch.tensor(np.zeros((self.num_layers, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            h_0 = data_input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0 = Variable(h_0)

        h_0 = h_0.to(self.device)

        # xs_embedding.shape: (seq, batch, hidden_dim)
        xs_embedding = self.embeddings(data_input)

        pre, h_0 = self.lstm(xs_embedding, h_0)
        # hidden_drop = self.dropout(hidden)
        # flatten_hidden = self.flatten(hidden_drop)
        # pre = self.linear(flatten_hidden)

        # ((seq_len * batch_size),hidden_dim)

        # Flatten操作后进入Linear层
        # pre = self.linear(nn.Flatten()(pre))
        pre = self.linear(pre.view(seq_len * batch_size, -1))
        return pre, h_0
