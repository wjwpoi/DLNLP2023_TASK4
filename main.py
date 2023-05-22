import jieba
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import re
import random


def get_text(path, index):
    with open(path + index, encoding='gbk', errors='ignore') as _index:
        if path == 'data1/':
            _index = _index.readline()
            books = _index.strip().split(',')
        else:
            books = _index.readlines()
            books = [book.strip('\n') for book in books]

    all_text = []  # content, label
    print("# 正在提取语料")
    for book in tqdm(books):
        with open(path + book + '.txt', encoding='gbk', errors='ignore') as f:
            txt = f.read()
            txt = re.sub('[。!?”](\\n\\u3000\\u3000)+', '注意这里要分段', txt)
            txt = re.sub('[\\sa-zA-Z0-9]+', '', txt)
            txt = txt.split('注意这里要分段')
            all_text += txt
    return all_text


class Dictionary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1


class Corpus(object):

    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, all_text, batch_size=32, mode='char'):
        print('# 正在创建词典')
        tokens = 0
        EOS = '<eos>'
        self.dictionary.add_word(EOS)
        for seq in all_text:
            if mode == 'word':
                seq = jieba.lcut(seq)
            tokens += len(seq) + 1
            for word in seq:
                self.dictionary.add_word(word)

        print('# 正在转换为序列(最后阶段会相对很慢)')
        token = 0
        ids = torch.LongTensor(tokens)
        for seq in tqdm(all_text):
            if mode == 'word':
                seq = jieba.cut(seq)
            for word in seq:
                ids[token] = self.dictionary.word2idx[word]
                token += 1
            ids[token] = self.dictionary.word2idx[EOS]
            token += 1

        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids



class MyLSTM(nn.Module):

    def __init__(self, dictionary, vocab_size, embed_size=128, hidden_size=1024, num_layers=1):
        super(MyLSTM, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dictionary = dictionary
        self.num_layers = num_layers

    def forward(self, x, h):
        x = self.word_embedding(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)

        return out, (h, c)

    def generate(self, _input, max_len=300):

        state = (torch.zeros(self.num_layers, 1, self.hidden_size).to(device),
                 torch.zeros(self.num_layers, 1, self.hidden_size).to(device))
        article = ''

        for word in _input[:-1]:
            word = self.str2tensor(word)
            _, state = self.forward(word, state)

        _input = self.str2tensor(_input[-1])
        for i in range(max_len):
            output, state = self.forward(_input, state)

            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            _input.fill_(word_id)
            word = self.dictionary.idx2word[word_id]
            if word == '<eos>':
                return article
            article += word

        return article

    def cal_score(self, seq):
        seq = torch.LongTensor([self.str2tensor(word) for word in seq]).to(device)
        seq_vec = self.word_embedding(seq)
        return torch.mean(seq_vec, dim=0)

    def cal_score_between_seqs(self, seq1, seq2):
        score1 = self.cal_score(seq1)
        score2 = self.cal_score(seq2)
        score = torch.cosine_similarity(score1, score2, dim=0)
        return score

    def str2tensor(self, word):
        word = torch.tensor(self.dictionary.word2idx[word], device=device).long()
        word = word.unsqueeze(-1).unsqueeze(-1)
        return word


if __name__ == "__main__":
    train = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jinyong = get_text('data1/', 'inf.txt')
    gulong = get_text('data2/', '古龙武侠小说全集目录.txt')
    all_text = jinyong + gulong

    if train:
        embed_size = 256
        hidden_size = 1024
        num_layers = 1
        num_epochs = 10
        batch_size = 512
        seq_length = 256

        corpus = Corpus()
        ids = corpus.get_data(all_text, batch_size)
        vocab_size = len(corpus.dictionary)
        model = MyLSTM(corpus.dictionary, vocab_size, embed_size, hidden_size, num_layers).to(device)
        model_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        print("# 开始训练")
        for epoch in tqdm(range(num_epochs)):
            states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                      torch.zeros(num_layers, batch_size, hidden_size).to(device))

            print("# epoch " + str(epoch) + " start.")
            for i in range(0, ids.size(1) - seq_length, seq_length):
                inputs = ids[:, i:i + seq_length].to(device)
                targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)

                states = [state.detach() for state in states]
                outputs, states = model(inputs, states)
                loss = model_loss(outputs, targets.reshape(-1))

                model.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        torch.save(model, 'model.pt')

    print("# 开始测试")
    model = torch.load('model.pt')
    max_len = 500

    with open('res.txt', 'w') as f:
        mean_score = 0
        nums = 200
        for i in range(200):
            j = random.randint(0, len(all_text)-2)
            _input = all_text[j]
            _output = all_text[j+1]
            article = model.generate(_input, max_len)
            f.write(str(i) + ' 原文：' + _input + '\n')
            f.write('真实：' + _output + '\n')
            f.write('生成：' + article + '\n')
            score = float(model.cal_score_between_seqs(_output, article))
            f.write('score：' + str(score) + '\n')
            if article == '':
                continue
            mean_score += score
        mean_score /= 200
        print('平均评分：' + str(mean_score))
