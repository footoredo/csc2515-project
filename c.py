from collections import defaultdict
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import joblib
import numpy as np
from datetime import datetime

import multiprocessing as mp

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from a import TEXT_LIMIT

from common import *


# EMBEDDING_DIM = 100
# TEXT_LIMIT = 200
# HIDDEN_DIM = 128
# USER_SIZE = 12186 + 1
# BATCH_SIZE = 64
# NUM_LAYERS = 1
# DROPOUT = 0.2


EMBEDDING_DIM = 50
TEXT_LIMIT = 200
HIDDEN_DIM = 64
USER_SIZE = 12186 + 1
BATCH_SIZE = 200
NUM_LAYERS = 2
DROPOUT = 0.2


class Dataset:
    
    def __init__(self, filename, device):
        self.device = device
        glove_vec, glove_idx_map, glove_words = load_twitter_glove(EMBEDDING_DIM)
        print(glove_vec.shape, glove_vec.dtype)
        # return
        raw_data = joblib.load(data_dir / filename)
        n = len(raw_data['id'])
        user_idx_dict = joblib.load(data_dir / "user_idx.obj")
        
        self.raw_data = raw_data
        # self.price_checker = price_checker
        self.user_idx_dict = user_idx_dict
        self.glove_vec = glove_vec
        self.glove_idx_map = glove_idx_map
        self.glove_words = glove_words
        
        print("Total data number:", n)
        
        st0 = time.time()
        
        # with Pool(8) as p:
        #     data = p.map(self._convert_data, range(n))
        #     text_idx = []
        #     user_idx = []
        #     extra_info = []
        #     price = []
        #     self.n = 0
        #     for i in range(n):
        #         if data[i] is not None:
        #             price.append(data[i][0])
        #             text_idx.append(data[i][1])
        #             user_idx.append(data[i][2])
        #             extra_info.append(data[i][3])
        #             self.n += 1
        
        n_processes = 8
        chunk_size = (n + n_processes - 1) // n_processes
        ps = []
        data_queue = mp.Queue()
        for i in range(n_processes):
            l = i * chunk_size
            r = min((i + 1) * chunk_size, n)
            p = mp.Process(target=self._convert_data, 
                           args=(i, data_queue, raw_data["timestamp"][l:r], raw_data["text"][l:r],
                                 raw_data["user"][l:r], raw_data["replies"][l:r], raw_data["likes"][l:r],
                                 raw_data["retweets"][l:r]))
            ps.append(p)
            p.start()
            
        text_idx = []
        user_idx = []
        extra_info = []
        price = []
        n_finished = 0
        self.n = 0
        while n_finished < n_processes:
            data = data_queue.get()
            if data is None:
                n_finished += 1
                print(n_finished, flush=True)
            else:
                _text_idx, _user_idx, _extra_info, _price = data
                text_idx.append(_text_idx)
                user_idx.append(_user_idx)
                extra_info.append(_extra_info)
                price.append(_price)
                self.n += 1
                
        for p in ps:
            p.join()
        
        # idx_data = [None] * n_processes
        # while not data_queue.empty():
        #     data = data_queue.get()
        #     idx_data[data[0]] = data[1:]
        
        # for _text_idx, _user_idx, _extra_info, _price in idx_data:
        #     text_idx += _text_idx
        #     user_idx += _user_idx
        #     extra_info += _extra_info
        #     price += _price
        
        # for i in range(n):
            
            # if (i + 1) % 10000 == 0:
            #     t = time.time()
            #     print(f"#{i}, time: {t - st0:.2f}s, timedelta: {t - st:.2f}s", flush=True)
            #     st = t
        
            
        print(f"Total loading time: {time.time() - st0:.2f}s.", flush=True)
        
        self.data = {
            "text_idx": torch.tensor(np.array(text_idx)).to(device),
            "user_idx": torch.LongTensor(user_idx).to(device),
            "extra_info": torch.tensor(np.array(extra_info), dtype=torch.float).to(device),
            "price": torch.tensor(price, dtype=torch.float).unsqueeze(-1).to(device)
        }
        self.ordered_data_keys = ["text_idx", "user_idx", "extra_info", "price"]
        self.cur = 0
        
    def _convert_data(self, idx, data_queue, timestamps, texts, users, replies, likes, retweets):
        price_checker = PriceChecker()
        n = len(timestamps)
        # text_idx = []
        # user_idx = []
        # extra_info = []
        # price = []
        st = time.time()
        for i in range(n):
            if (i + 1) % 10000 == 0:
                print(idx, i, f"{time.time() - st:.2f}s", flush=True)
            
            ts = timestamps[i]
            d0 = datetime.fromtimestamp(ts)
            _price = price_checker.get_trend(d0, 3)
            if _price is None:
                continue
            
            # price.append(_price)
            
            words = twitter_preprocess(texts[i]).split()
            _text_idx = np.zeros(TEXT_LIMIT, dtype=np.int64)
            
            for i in range(TEXT_LIMIT):
                if i < len(words):
                    word = words[i]
                    if word not in self.glove_idx_map:
                        word = "<unknown>"
                else:
                    word = "<pad>"
                _text_idx[i] = self.glove_idx_map[word]
            
            # text_idx.append(_text_idx)
            
            try:
                # user_idx = self.user_idx_dict[text_data['user'][i]] + 100004
                _user_idx = self.user_idx_dict[users[i]] + 1
            except KeyError:
                # user_idx = 100003
                _user_idx = 0
            # user_idx.append(_user_idx)
            
            _extra_info = np.array([d0.hour * 60 + d0.minute, replies[i], likes[i], retweets[i]])
            
            data_queue.put((_text_idx, _user_idx, _extra_info, _price))
            
            # price.append(_price)
            # text_idx.append(_text_idx)
            # user_idx.append(_user_idx)
            # extra_info.append(_extra_info)
            # return _price, _text_idx, _user_idx, _extra_info
        # data_queue.put((idx, text_idx, user_idx, extra_info, price))
        print(idx, "finished", flush=True)
        data_queue.put(None)
        
    @property            
    def vocab_size(self):
        return len(self.glove_words)
    
    # def __getattr__(self, key):
    #     return self.data[key]
    
    def shuffle(self):
        indices = torch.randperm(self.n).to(self.device)
        for key in self.data.keys():
            self.data[key] = self.data[key][indices]
        
    def iter_batch(self, batch_size, shuffle=True, stop=True):
        # if shuffle:
        #     self.shuffle()
        while True:
            r = min(self.cur + batch_size, self.n)
            yield [self.data[key][self.cur: r] for key in self.ordered_data_keys]
            self.cur = r
            if self.cur == self.n:
                if shuffle:
                    self.shuffle()
                self.cur = 0
                if stop:
                    break
              
              
class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, extra_input_dim, hidden_dim, vocab_size, user_size, embedding_weights):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weights is not None:
            self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_weights, dtype=torch.float), requires_grad=True)
        self.user_embeddings = nn.Embedding(user_size, 64)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=True)
        # self.drop = nn.Dropout(p=0.5)
        # self.lstmw = nn.Linear(hidden_dim, hidden_dim)
        # self.lstmln = nn.LayerNorm(hidden_dim)

        self.extraw = nn.Linear(64 + extra_input_dim, hidden_dim)
        self.extra2gate = nn.Linear(hidden_dim, 1)
        # self.extraln = nn.LayerNorm(hidden_dim)
        self.hidden2scalar = nn.Linear(hidden_dim * 3, 1)

    def forward(self, sentence, user_idx, extra_inpts):
        embeds = self.word_embeddings(sentence)
        # print("embeds:", embeds.shape)
        # print(embeds[0].detach().cpu().numpy())
        # print(embeds[1].detach().cpu().numpy())
        # print(embeds.shape, extra_inpts.shape)
        # inpt = torch.cat([embeds, extra_inpts], -1)
        inpt = embeds
        lstm_out, _ = self.lstm(inpt)
        lstm_out = torch.cat([lstm_out[:, 0, self.hidden_dim:], lstm_out[:, -1, :self.hidden_dim]], 1)
        # print("lstm_out:", lstm_out.shape)
        # print(lstm_out[0].detach().cpu().numpy())
        # print(lstm_out[1].detach().cpu().numpy())
        # lstm_out = F.relu(self.lstmw(lstm_out))
        lstm_out = F.relu(lstm_out)
        
        # print(scalar_out[0].detach().cpu().numpy())
        # print(scalar_out[1].detach().cpu().numpy())

        user_embeds = self.user_embeddings(user_idx)
        extras = torch.cat([user_embeds, extra_inpts], -1)
        extras = F.relu(self.extraw(extras))
        # gate = torch.sigmoid(self.extra2gate(extras))
        
        # print(scalar_out.shape)
        # return torch.sigmoid(scalar_out)
        
        out = torch.cat([lstm_out, extras], 1)
        # out = lstm_out
        scalar_out = self.hidden2scalar(out)
        
        return scalar_out


def eval(epoch, batch, model, dataset, max_batch=10):
    epoch_loss = 0
    epoch_acc = 0
    epoch_n = 0
    for _batch, (text_idx, user_idx, extra_info, price) in enumerate(dataset.iter_batch(200, shuffle=True, stop=False)):
        if _batch >= max_batch:
            break
        price_pred = model(text_idx, user_idx, extra_info)
        loss = (0.5 * (price_pred - price) ** 2).detach().cpu().numpy()
        acc = (torch.sign(price_pred) == torch.sign(price)).float().detach().cpu().numpy()
        batch_size = loss.shape[0]
        
        epoch_loss += loss.mean() * batch_size
        epoch_acc += acc.mean() * batch_size
        epoch_n += batch_size
    
    epoch_loss /= epoch_n
    epoch_acc /= epoch_n
    
    test_info = { 'epoch': epoch, 'batch': batch, 'loss': epoch_loss, 'acc': epoch_acc }
    
    return test_info

              
def train(run):
    device = torch.device("cuda:0") 
    dataset = Dataset("raw_data-1m.obj", device)
    eval_dataset = Dataset("raw_data-test-mini.obj", device)
    
    model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, dataset.vocab_size, USER_SIZE, dataset.glove_vec)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    
    infos = []
    train_infos = []
    eval_infos = []

    criterion = nn.BCELoss()

    print("start training")
    
    sum_batch = 0

    for epoch in range(10):
        st = time.time()
        # for batch, (text_idx, text_v, price, idx) in enumerate(dataloader):
        epoch_loss = 0
        epoch_acc = 0
        epoch_n = 0
        for batch, (text_idx, user_idx, extra_info, price) in enumerate(dataset.iter_batch(BATCH_SIZE)):
            # text_idx = text_idx.to(device)
            # text_v = text_v.to(device)
            # price = price.to(device)

            # if batch == 26:
            #     for i in range(100):
            #         print(text_idx[i])
            #         print(text_v[i])
            #         print(price[i])

            _price = (price > 0).float()
            optimizer.zero_grad()

            price_pred = model(text_idx, user_idx, extra_info)
            if batch == 0:
                if epoch == 0:
                    print("text_idx", text_idx.shape)
                    print("user_idx", user_idx.shape)
                    print("extra_info", extra_info.shape)
                    print("price", price.shape)
                    print(text_idx[0].cpu().numpy())
                    print(text_idx[1].cpu().numpy())
                print(price_pred[:20, 0].detach().cpu().numpy())
                print(price[:20, 0].cpu().numpy())
            loss = (0.5 * (price_pred - price) ** 2).mean()
            # loss = criterion(price_pred, _price)
            # acc = ((price_pred > 0).float() - _price).abs().mean()
            acc = (torch.sign(price_pred) == torch.sign(price)).float().mean()

            loss.backward()
            optimizer.step()

            info = { 'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'acc': acc.item() }
            epoch_loss += loss.item() * text_idx.shape[0]
            epoch_acc += acc.item() * text_idx.shape[0]
            epoch_n += text_idx.shape[0]
            infos.append(info)

            print(info, f"{time.time() - st:.2f}s", flush=True)
            st = time.time()

            if np.isnan(loss.item()):
                return
            
            if sum_batch % 100 == 0:
                eval_info = eval(epoch, batch, model, eval_dataset, max_batch=10)
                print('eval', eval_info, flush=True)
                eval_infos.append(eval_info)
            sum_batch += 1
            
        train_info = { 'epoch': epoch, 'loss': epoch_loss / epoch_n, 'acc': epoch_acc / epoch_n }
        train_infos.append(train_info)
        print("train:", train_info)
        torch.save(model.state_dict(), str(data_dir / f"model-c-{run}-{epoch}.pkl"))
        joblib.dump(infos, data_dir / f"infos-c-{run}.data")
        joblib.dump(train_infos, data_dir / f"train_infos-{run}.data")
        joblib.dump(eval_infos, data_dir / f"eval_infos-{run}.data")
        
    
def test(run):
    device = torch.device("cuda:0") 
    dataset = Dataset("raw_data-test-mini.obj", device)
    
    model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, dataset.vocab_size, USER_SIZE, dataset.glove_vec)
    model.to(device)
    
    user_idx_dict = dataset.user_idx_dict
    r_user_idx_dict = dict()
    for user, idx in user_idx_dict.items():
        r_user_idx_dict[idx + 1] = user
    r_user_idx_dict[0] = "<unknown>"
    
    test_infos = []
    
    for epoch in range(100):
        try:
            model.load_state_dict(torch.load(str(data_dir / f"model-c-{run}-{epoch}.pkl")))
        except:
            break
        epoch_loss = 0
        epoch_acc = 0
        epoch_n = 0
        user_loss = defaultdict(float)
        user_acc = defaultdict(float)
        user_n = defaultdict(int)
        for batch, (text_idx, user_idx, extra_info, price) in enumerate(dataset.iter_batch(200)):
            price_pred = model(text_idx, user_idx, extra_info)
            loss = (0.5 * (price_pred - price) ** 2).detach().cpu().numpy()
            acc = (torch.sign(price_pred) == torch.sign(price)).float().detach().cpu().numpy()
            batch_size = loss.shape[0]
            
            epoch_loss += loss.mean() * batch_size
            epoch_acc += acc.mean() * batch_size
            epoch_n += batch_size
            
            for i in range(batch_size):
                user = r_user_idx_dict[user_idx[i].item()]
                user_loss[user] += loss[i]
                user_acc[user] += acc[i]
                user_n[user] += 1
        
        epoch_loss /= epoch_n
        epoch_acc /= epoch_n
        
        test_info = { 'epoch': epoch, 'loss': epoch_loss, 'acc': epoch_acc }
        print(test_info, flush=True)
        
        for user in user_n.keys():
            user_loss[user] /= user_n[user]
            user_acc[user] /= user_n[user]
            
        print("Top 20 low loss users:")
        cnt = 0
        for user in sorted(user_n.keys(), key=user_loss.get):
            if user_n[user] >= 200:
                cnt += 1
                print(user, user_loss[user])
                if cnt == 20:
                    break
                
        print("Top 20 high acc users:")
        cnt = 0
        for user in sorted(user_n.keys(), key=user_acc.get, reverse=True):
            if user_n[user] >= 200:
                cnt += 1
                print(user, user_acc[user])
                if cnt == 20:
                    break
            
        test_info['user_loss'] = dict(user_loss)
        test_info['user_acc'] = dict(user_acc)
        
        if len(test_infos) == 0:
            test_info['user_n'] = dict(user_n)
        
        test_infos.append(test_info)
        joblib.dump(test_infos, data_dir / f"test_infos-{run}.data")
    
    
if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "test":
        test(sys.argv[2])