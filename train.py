import torch
import tqdm as tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import train_vec
from mymodel_lstm import MyLstmModel
from torchnet import meter

from mymodel_rnn import MyRnnModel

device = "cuda" if torch.cuda.is_available() else "cpu"


# 数据预处理的过程都是一样的
def get_data_args(batch_size):
    # train_vec->数据预处理：Json->txt，拆分，词向量
    data, (w1, word_2_index, index_2_word) = train_vec()

    # w1: (vocab_size, embedding_dim)
    # vocab_size = word_size : 词汇表"词库"的大小
    # embedding_dim = vector_size = embedding_num :词嵌入的维度
    vocab_size, embedding_dim = w1.shape

    # 数据封装 DaraLoader/Dataset
    # DataLoader批量加载数据 -> 数据在 DataLoader 对象中已经被封装好了
    # batch_size: 每个批次（batch）中包含的样本数量 -> 将batch_size个数据打包成一份
    # shuffle: 是否在每个 epoch 之前打乱数据集的顺序 ;False->不打乱
    # pin_memory : 是否将数据加载到 CUDA 的固定内(这里是CPU版的pytorch)
    # drop_last : 是否丢弃最后一个不完整的批次->确保每个批次都有相同数量的样本
    dataloader = DataLoader(data, batch_size, shuffle=True, drop_last=True)

    # shuffle=True则需要确保 __getitem__ 方法返回的数据长度是固定的，所有序列的长度要求一致
    return embedding_dim, vocab_size, index_2_word, dataloader


# 获取参数 在这里修改
def get_args():
    # (*)
    batch_size = 64

    # (*)hidden_num:
    hidden_dim = 130

    # (*)lr : 学习率
    lr = 1e-3
    # (*)训练过程中的轮数
    epochs = 40
    layers_num = 2

    return batch_size, hidden_dim, layers_num, lr, epochs


# 两个网络的训练过程都是类似的，只是使用的模型不一样
def train(type):
    # 获取参数
    batch_size, hidden_dim, layers_num, lr, epochs = get_args()

    # return: embedding_dim, vocab_size, index_2_word, dataloader
    embedding_dim, vocab_size, index_2_word, dataloader = get_data_args(batch_size)

    # 默认 -> 输入不存在的数字也训练的是使用LSTM层的网络
    model_file = './model/model_lstm'
    model = MyLstmModel(vocab_size, embedding_dim, hidden_dim, layers_num)

    if type == 3:
        return

    elif type == 1:
        model_file = './model/model_lstm'
        # MyLstmModel(vocab_size, embedding_dim, hidden_dim, num_layers)
        model = MyLstmModel(vocab_size, embedding_dim, hidden_dim, layers_num)

    elif type == 2:
        model_file = './model/model_rnn'
        model = MyRnnModel(vocab_size, embedding_dim, hidden_dim, layers_num)

    model = model.to(device)
    # 损失函数
    criterion = model.loss

    # (*)优化器，梯度更新
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 计算每个epoch的损失值
    loss_meter = meter.AverageValueMeter()
    period = []
    loss2 = []

    for e in range(epochs):
        # 清空之前记录的所有损失值数据
        loss_meter.reset()

        # sequence_length(一首古诗) : 32

        # 交的项目这里的注解写错了好多：batch_first改回False了，不是下面的结构
        # xs_embedding.shape: (sequence_length,batch_size,embedding_dim)
        # ys_index.shape: (sequence_length,batch_size)

        # 因为一开始使用的MyDataset:Dataset实现的数据封装，忘记改了所以注解写错了
        # 迭代 DataLoader 对象
        for batch_index, data in tqdm.tqdm(enumerate(dataloader)):
            # 维度转置+存储连续
            data = data.long().transpose(0, 1).contiguous()
            data = data.to(device)

            optimizer.zero_grad()
            # target 为真值
            x_train, y_train = Variable(data[:-1, :]), Variable(data[1:, :])

            pre, _ = model(x_train)
            loss = criterion(pre, y_train.view(-1))
            # nn.utils.clip_grad_norm(model.parameters(), 1)

            # 计算更新
            loss.backward()
            optimizer.step()

            # 观察记录
            # loss_meter.add(loss.item())
            # period.append(batch_index + e * len(dataloader))
            # loss2.append(loss_meter.value()[0])

            # 定期打印，观察
            if batch_index % 20 == 0:
                # print(loss)
                # 每20个epoch保存一次model
                torch.save(model.state_dict(), model_file)

    # 保存最终model
    torch.save(model.state_dict(), model_file)
