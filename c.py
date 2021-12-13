from collections import defaultdict
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import joblib
import numpy as np
from datetime import datetime

import multiprocessing as mp

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from a import TEXT_LIMIT, VOCAB_SIZE

from common import *


# EMBEDDING_DIM = 100
# TEXT_LIMIT = 200
# HIDDEN_DIM = 128
# USER_SIZE = 12186 + 1
# BATCH_SIZE = 64
# NUM_LAYERS = 1
# DROPOUT = 0.2


EMBEDDING_DIM = 100
TEXT_LIMIT = 200
HIDDEN_DIM = 128
USER_SIZE = 12186 + 1
BATCH_SIZE = 200
NUM_LAYERS = 2
DROPOUT = 0.2


def split_dataset(filename, train_ratio):
    raw_data = joblib.load(data_dir / f"{filename}.obj")
    n = len(raw_data['id'])
    n_train = int(n * train_ratio)
    n_test = n - n_train
    
    print(f"Total of {n} data points will be split into {n_train} train data points and {n_test} test data points.")
    
    indices = list(range(n))
    np.random.shuffle(indices)
    
    train_data = dict()
    test_data = dict()
    
    for key in raw_data.keys():
        train_data[key] = []
        test_data[key] = []
    
    for i in range(n_train):
        for key in raw_data.keys():
            train_data[key].append(raw_data[key][i])
            
    for i in range(n_train, n):
        for key in raw_data.keys():
            test_data[key].append(raw_data[key][i])
            
    joblib.dump(train_data, data_dir / f"{filename}.train.obj")
    joblib.dump(test_data, data_dir / f"{filename}.test.obj")


class Dataset:
    
    def __init__(self, filename, device):
        self.device = device
        glove_vec, glove_idx_map, glove_words = load_twitter_glove(EMBEDDING_DIM)
        print(glove_vec.shape, glove_vec.dtype)
        # return
        raw_data = joblib.load(data_dir / f"{filename}.obj")
        n = len(raw_data['id'])
        user_idx_dict = joblib.load(data_dir / "user_idx.obj")
        
        self.raw_data = raw_data
        # self.price_checker = price_checker
        self.user_idx_dict = user_idx_dict
        self.glove_vec = glove_vec
        self.glove_idx_map = glove_idx_map
        self.glove_words = glove_words
        
        self.r_word_idx_dict = dict()
        for word, idx in glove_idx_map.items():
            self.r_word_idx_dict[idx] = word
        
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
        
        n_processes = 12
        chunk_size = (n + n_processes - 1) // n_processes
        ps = []
        data_queue = mp.Queue()
        for i in range(n_processes):
            l = i * chunk_size
            r = min((i + 1) * chunk_size, n)
            p = mp.Process(target=self._convert_data, 
                           args=(i, data_queue, raw_data["timestamp"][l:r], raw_data["text"][l:r],
                                 raw_data["user"][l:r], raw_data["replies"][l:r], raw_data["likes"][l:r],
                                 raw_data["retweets"][l:r], list(range(l, r))))
            ps.append(p)
            p.start()
            
        text_idx = []
        text_length = []
        user_idx = []
        extra_info = []
        price = []
        idx = []
        n_finished = 0
        self.n = 0
        while n_finished < n_processes:
            data = data_queue.get()
            if data is None:
                n_finished += 1
                print(n_finished, flush=True)
            else:
                _text_idx, _text_length, _user_idx, _extra_info, _price, _idx = data
                text_idx.append(_text_idx)
                text_length.append(_text_length)
                user_idx.append(_user_idx)
                extra_info.append(_extra_info)
                price.append(_price)
                idx.append(_idx)
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
        
            
        print(f"Total valid data points: {self.n}.", flush=True)
        print(f"Total loading time: {time.time() - st0:.2f}s.", flush=True)
        
        self.data = {
            "text_idx": torch.tensor(np.array(text_idx)).to(device),
            "user_idx": torch.LongTensor(user_idx).to(device),
            "extra_info": torch.tensor(np.array(extra_info), dtype=torch.float).to(device),
            "price": torch.tensor(price, dtype=torch.float).unsqueeze(-1).to(device),
            "text_length": torch.LongTensor(text_length),
            "idx": torch.LongTensor(idx)
        }
        self.ordered_data_keys = ["text_idx", "text_length", "user_idx", "extra_info", "price", "idx"]
        self.cur = 0
        
    def _convert_data(self, process_idx, data_queue, timestamps, texts, users, replies, likes, retweets, indices):
        price_checker = PriceChecker()
        n = len(timestamps)
        # text_idx = []
        # user_idx = []
        # extra_info = []
        # price = []
        st = time.time()
        for i in range(n):
            if (i + 1) % 10000 == 0:
                print(process_idx, i, f"{time.time() - st:.2f}s", flush=True)
            
            ts = timestamps[i]
            d0 = datetime.fromtimestamp(ts)
            # _price = price_checker.get_trend(d0, 3)
            # _price = price_checker.get_trend(d0, 5, only_past=True)
            _price = price_checker.get_trend(d0, 2, only_future=True)
            if _price is None:
                continue
            
            # price.append(_price)
            
            words = twitter_preprocess(texts[i]).split()
            _text_idx = np.zeros(TEXT_LIMIT, dtype=np.int64)
            
            for j in range(TEXT_LIMIT):
                if j < len(words):
                    word = words[j]
                    if word not in self.glove_idx_map:
                        word = "<unknown>"
                else:
                    word = "<pad>"
                _text_idx[j] = self.glove_idx_map[word]
                
            _text_length = min(TEXT_LIMIT, len(words))
            
            if _text_length == 0:
                continue
            
            # text_idx.append(_text_idx)
            
            try:
                # user_idx = self.user_idx_dict[text_data['user'][i]] + 100004
                _user_idx = self.user_idx_dict[users[i]] + 1
            except KeyError:
                # user_idx = 100003
                _user_idx = 0
            # user_idx.append(_user_idx)
            
            _extra_info = np.array([d0.hour * 60 + d0.minute, replies[i], likes[i], retweets[i]])
            
            data_queue.put((_text_idx, _text_length, _user_idx, _extra_info, _price, indices[i]))
            
            # price.append(_price)
            # text_idx.append(_text_idx)
            # user_idx.append(_user_idx)
            # extra_info.append(_extra_info)
            # return _price, _text_idx, _user_idx, _extra_info
        # data_queue.put((idx, text_idx, user_idx, extra_info, price))
        print(process_idx, "finished", flush=True)
        data_queue.put(None)
        
    @property            
    def vocab_size(self):
        return len(self.glove_words)
    
    # def __getattr__(self, key):
    #     return self.data[key]
    
    def get_raw_data(self, idx):
        data = dict()
        for key in self.raw_data.keys():
            data[key] = self.raw_data[key][idx]
        return data
    
    def shuffle(self):
        indices = torch.randperm(self.n).to(self.device)
        for key in self.data.keys():
            self.data[key] = self.data[key][indices]
        
    def iter_batch(self, batch_size, shuffle=True, stop=True, reset=True):
        # if shuffle:
        #     self.shuffle()
        if reset:
            if shuffle:
                self.shuffle()
            self.cur = 0
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
        user_embed_size = 64

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weights is not None:
            self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_weights, dtype=torch.float), requires_grad=True)
        self.user_embeddings = nn.Embedding(user_size, user_embed_size)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=True)
        # self.drop = nn.Dropout(p=0.5)
        # self.lstmw = nn.Linear(hidden_dim, hidden_dim)
        self.lstmln = nn.LayerNorm(hidden_dim * 2)


        # # old
        # self.extraw = nn.Linear(user_embed_size + extra_input_dim, hidden_dim)
        # self.extra2gate = nn.Linear(hidden_dim, 1)
        # self.extraln = nn.LayerNorm(hidden_dim)
        # # self.hiddenln = nn.LayerNorm(hidden_dim * 3)
        # self.hidden2scalar = nn.Linear(hidden_dim * 3, 1)
        # self.hidden2gate = nn.Linear(hidden_dim * 3, 1)
        
        # new
        
        self.userw = nn.Linear(user_embed_size, hidden_dim)

        self.extraw = nn.Linear(extra_input_dim, hidden_dim)
        self.extra2gate = nn.Linear(hidden_dim * 2, 1)
        self.extraln = nn.LayerNorm(hidden_dim * 2)
        # self.hiddenln = nn.LayerNorm(hidden_dim * 3)
        self.hidden2scalar = nn.Linear(hidden_dim * 4, 1)
        
        self.lstm2gate = nn.Linear(hidden_dim * 2, 1)
        self.user2gate = nn.Linear(hidden_dim, 1)
        self.extra2gate = nn.Linear(hidden_dim, 1)

    def forward(self, sentence, sentence_length, user_idx, extra_inpts):
        embeds = self.word_embeddings(sentence)
        # print("embeds:", embeds.shape)
        # print(embeds[0].detach().cpu().numpy())
        # print(embeds[1].detach().cpu().numpy())
        # print(embeds.shape, extra_inpts.shape)
        # inpt = torch.cat([embeds, extra_inpts], -1)
        inpt = pack_padded_sequence(embeds, sentence_length, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(inpt)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = torch.cat([lstm_out[:, 0, self.hidden_dim:], lstm_out[range(len(lstm_out)), sentence_length - 1, :self.hidden_dim]], 1)
        # print("lstm_out:", lstm_out.shape)
        # print(lstm_out[0].detach().cpu().numpy())
        # print(lstm_out[1].detach().cpu().numpy())
        # lstm_out = F.relu(self.lstmw(lstm_out))
        
        # lstm_out = F.relu(self.lstmln(lstm_out))
        lstm_out = F.relu(lstm_out)
        
        # print(scalar_out[0].detach().cpu().numpy())
        # print(scalar_out[1].detach().cpu().numpy())


        # # old
        # user_embeds = self.user_embeddings(user_idx)
        # extra = torch.cat([user_embeds, extra_inpts], -1)
        # # extra = F.relu(self.extraln(self.extraw(extra)))
        # extra = F.relu(self.extraw(extra))
        
        # # print(scalar_out.shape)
        # # return torch.sigmoid(scalar_out)
        
        # out = torch.cat([lstm_out, extra], 1)
        # # out = lstm_out
        # gate = torch.sigmoid(self.hidden2gate(out)) * 0.99 + 0.01
        # # gate = torch.sigmoid(self.hidden2gate(extra)) * 0.99 + 0.01
        
        
        # new
        user_embeds = self.user_embeddings(user_idx)
        user_hidden = F.relu(self.userw(user_embeds))
        extra_hidden = F.relu(self.extraw(extra_inpts))
        
        lstm_gate = torch.sigmoid(self.lstm2gate(lstm_out))
        user_gate = torch.sigmoid(self.user2gate(user_hidden))
        extra_gate = torch.sigmoid(self.extra2gate(extra_hidden))
        # gate = lstm_gate * extra_gate * 0.99 + 0.01
        gate = lstm_gate * user_gate * extra_gate * 0.99 + 0.01
        
        extra = torch.cat([user_hidden, extra_hidden], -1)
        
        out = torch.cat([lstm_out, extra], 1)
        
        
        scalar_out = self.hidden2scalar(out)
        
        return scalar_out, gate
    
    
def display(dataset, idx, text_idx=None, show_raw_text=False):
    data = dataset.get_raw_data(idx)
    print(f"user: {data['user']}")
    
    if show_raw_text:
        print("--- raw text ---")
        print(data['text'])
        
    if text_idx is not None:
        print("--- processed text ---")
        for j in range(TEXT_LIMIT):
            _text_idx = text_idx[j].item()
            if _text_idx == dataset.vocab_size - 1:  # pad
                break
            print(dataset.r_word_idx_dict[_text_idx], end=" ")
        print()
    # minutes = int(extra_info[i, 0].item())
    # replies = int(extra_info[i, 1].item())
    # likes = int(extra_info[i, 2].item())
    # retweets = int(extra_info[i, 3].item())
    replies = data['replies']
    likes = data['likes']
    retweets = data['retweets']
    print(f"{datetime.fromtimestamp(data['timestamp']).isoformat()}, replies={replies}, likes={likes}, retweets={retweets}")


def eval(epoch, batch, sum_batch, model, dataset, shuffle=True, max_batch=None, user_stats=False, verbose=False):
    epoch_loss = 0
    epoch_acc = 0
    epoch_n = 0
    
    if user_stats or verbose:
        user_idx_dict = dataset.user_idx_dict
        r_user_idx_dict = dict()
        for user, idx in user_idx_dict.items():
            r_user_idx_dict[idx + 1] = user
        r_user_idx_dict[0] = "<unknown>"
        
        user_loss = defaultdict(float)
        user_tp = defaultdict(int)
        user_tn = defaultdict(int)
        user_fp = defaultdict(int)
        user_fn = defaultdict(int)
        user_weights = defaultdict(float)
        user_n = defaultdict(int)
    
    if max_batch is None:
        batch_generator = dataset.iter_batch(200, shuffle=shuffle)
    else:
        batch_generator = dataset.iter_batch(200, shuffle=shuffle, stop=False, reset=False)
        
    for _batch, (text_idx, text_length, user_idx, extra_info, price, idx) in enumerate(batch_generator):
        if max_batch is not None and _batch >= max_batch:
            break
        price_pred, weights = model(text_idx, text_length, user_idx, extra_info)
        all_loss = 0.5 * (price_pred - price) ** 2
        all_acc = (torch.sign(price_pred) == torch.sign(price)).float()
        loss = (all_loss * weights).sum() / weights.sum()
        acc = (all_acc * weights).sum() / weights.sum()
        batch_size = price.shape[0]
        
        epoch_loss += loss.item() * batch_size
        epoch_acc += acc.item() * batch_size
        epoch_n += batch_size
        
        if verbose:
            for i in range(batch_size):
                if dataset.get_raw_data(idx[i])['user'] == 'BitcoinInsight0':
                    print(f"\n#{i}, weights={weights[i].item()}, loss={all_loss[i].item()}, acc={all_acc[i].item()}")
                    print(f"trend={price[i].item()}, pred={price_pred[i].item()}")
                    display(dataset, idx[i], text_idx=text_idx[i], show_raw_text=True)
          
        if user_stats:
            all_loss = all_loss.detach().cpu().numpy()
            all_acc = all_acc.detach().cpu().numpy()
            weights = weights.detach().cpu().numpy()
            user_idx = user_idx.detach().cpu().numpy()
            for i in range(batch_size):
                user = dataset.get_raw_data(idx[i])['user']
                user_loss[user] += all_loss[i]
                sign_truth = np.sign(price[i].item())
                sign_pred = np.sign(price_pred[i].item())
                if sign_truth > 0:
                    if sign_pred > 0:
                        user_tp[user] += 1
                    else:
                        user_fn[user] += 1
                else:
                    if sign_pred > 0:
                        user_fp[user] += 1
                    else:
                        user_tn[user] += 1
                # user_acc[user] += all_acc[i]
                user_weights[user] += weights[i]
                user_n[user] += 1
    
    epoch_loss /= epoch_n
    epoch_acc /= epoch_n
    
    test_info = { 'epoch': epoch, 'batch': batch, 'sum_batch': sum_batch, 'loss': epoch_loss, 'acc': epoch_acc }
    
    if user_stats:
        user_acc = dict()
        user_prec = dict()
        user_recall = dict()
        user_f1 = dict()
        for user, n in user_n.items():
            user_loss[user] /= n
            user_weights[user] /= n
            
            tp, tn, fp, fn = user_tp[user], user_tn[user], user_fp[user], user_fn[user]
            assert tp + tn + fp + fn == n
            
            user_acc[user] = (tp + tn) / (tp + tn + fp + fn)
            user_prec[user] = tp / (tp + fp) if tp + fp > 0 else -1
            user_recall[user] = tp / (tp + fn) if tp + fn > 0 else -1
            user_f1[user] = 2 * (user_prec[user] * user_recall[user]) / (user_prec[user] + user_recall[user]) if user_prec[user] > 0 and user_recall[user] > 0 else -1
        
        return test_info, (user_loss, user_tp, user_tn, user_fp, user_fn, user_acc, user_prec, user_recall, user_f1, user_weights, user_n)
    else:
        return test_info

              
def train(run):
    device = torch.device("cuda:0") 
    dataset = Dataset("raw_data-1m.train", device)
    eval_dataset = Dataset("raw_data-1m.test", device)
    
    model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, dataset.vocab_size, USER_SIZE, dataset.glove_vec)
    # model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, dataset.vocab_size, USER_SIZE, None)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    
    infos = []
    train_infos = []
    eval_infos = []
    epoch_eval_infos = []

    criterion = nn.BCELoss()

    print("start training")
    
    sum_batch = 0
    
    epoch_eval_info = eval(0, 0, 0, model, eval_dataset, max_batch=None)
    print('epoch eval', epoch_eval_info, flush=True)
    epoch_eval_infos.append(epoch_eval_info)

    for epoch in range(100):
        st = time.time()
        # for batch, (text_idx, text_v, price, idx) in enumerate(dataloader):
        epoch_loss = 0
        epoch_acc = 0
        epoch_n = 0
        for batch, (text_idx, text_length, user_idx, extra_info, price, idx) in enumerate(dataset.iter_batch(BATCH_SIZE)):
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

            price_pred, weights = model(text_idx, text_length, user_idx, extra_info)
            if batch == 0:
                if epoch == 0:
                    print("text_idx", text_idx.shape)
                    print("user_idx", user_idx.shape)
                    print("extra_info", extra_info.shape)
                    print("price", price.shape)
                    print(text_idx[0].cpu().numpy())
                    print(text_idx[1].cpu().numpy())
                print("weights:", weights[:20, 0].detach().cpu().numpy())
                print("pred:", price_pred[:20, 0].detach().cpu().numpy())
                print("truth:", price[:20, 0].cpu().numpy())
            loss = ((0.5 * (price_pred - price) ** 2) * weights).sum() / weights.sum() - weights.mean()
            # loss = criterion(price_pred, _price)
            # acc = ((price_pred > 0).float() - _price).abs().mean()
            acc = ((torch.sign(price_pred) == torch.sign(price)).float() * weights).sum() / weights.sum()

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
            
            sum_batch += 1
            if sum_batch % 100 == 0:
                eval_info = eval(epoch, batch, sum_batch, model, eval_dataset, max_batch=10)
                print('eval', eval_info, flush=True)
                eval_infos.append(eval_info)
            
        train_info = { 'epoch': epoch, 'loss': epoch_loss / epoch_n, 'acc': epoch_acc / epoch_n }
        train_infos.append(train_info)
        print("train:", train_info)
        torch.save(model.state_dict(), str(data_dir / f"model-c-{run}-{epoch}.pkl"))
        joblib.dump(infos, data_dir / f"infos-c-{run}.data")
        joblib.dump(train_infos, data_dir / f"train_infos-{run}.data")
        joblib.dump(eval_infos, data_dir / f"eval_infos-{run}.data")
        
        epoch_eval_info = eval(epoch, 0, 0, model, eval_dataset, max_batch=None)
        print('epoch eval', epoch_eval_info, flush=True)
        epoch_eval_infos.append(epoch_eval_info)
        joblib.dump(epoch_eval_infos, data_dir / f"epoch_eval_infos-{run}.data")
        
    
def test(run):
    device = torch.device("cuda:0") 
    dataset = Dataset("raw_data-1m.test", device)
    # dataset = Dataset("raw_data-test-mini", device)
    
    model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, dataset.vocab_size, USER_SIZE, None)
    model.to(device)
    
    test_infos = []
    
    for epoch in range(20, 21):
        try:
            model.load_state_dict(torch.load(str(data_dir / f"model-c-{run}-{epoch}.pkl")))
        except:
            break
        
        test_info, user_info = eval(epoch, 0, 0, model, dataset, shuffle=False, max_batch=None, user_stats=True, verbose=True)
        user_loss, user_tp, user_tn, user_fp, user_fn, user_acc, user_prec, user_recall, user_f1, user_weights, user_n = user_info

        print(test_info, flush=True)
            
        threshold = 100
            
        print("Top 20 low loss users:")
        cnt = 0
        for user in sorted(user_n.keys(), key=user_loss.get):
            if user_n[user] >= threshold:
                cnt += 1
                print(user, user_loss[user], user_n[user])
                if cnt == 20:
                    break
                
        print("Top 20 high acc users:")
        cnt = 0
        for user in sorted(user_n.keys(), key=user_acc.get, reverse=True):
            if user_n[user] >= threshold:
                cnt += 1
                print(user, f"acc: {user_acc[user]}, prec: {user_prec[user]}, recall: {user_recall[user]}, f1: {user_f1[user]}", user_n[user])
                print(f"tp: {user_tp[user]}, tn: {user_tn[user]}, fp: {user_fp[user]}, fn: {user_fn[user]}")
                print(f"user positive: {user_tp[user] + user_fn[user]}, user negative: {user_tn[user] + user_fp[user]}")
                if cnt == 20:
                    break
                
        print("Top 20 high f1 users:")
        cnt = 0
        for user in sorted(user_n.keys(), key=user_f1.get, reverse=True):
            if user_n[user] >= threshold:
                cnt += 1
                print(user, f"acc: {user_acc[user]}, prec: {user_prec[user]}, recall: {user_recall[user]}, f1: {user_f1[user]}", user_n[user])
                print(f"tp: {user_tp[user]}, tn: {user_tn[user]}, fp: {user_fp[user]}, fn: {user_fn[user]}")
                print(f"user positive: {user_tp[user] + user_fn[user]}, user negative: {user_tn[user] + user_fp[user]}")
                if cnt == 20:
                    break
                
        print("Top 20 high weight users:")
        cnt = 0
        for user in sorted(user_n.keys(), key=user_weights.get, reverse=True):
            if user_n[user] >= threshold:
                cnt += 1
                print(user, user_weights[user], user_n[user])
                if cnt == 20:
                    break
            
        test_info['user_loss'] = dict(user_loss)
        test_info['user_acc'] = dict(user_acc)
        test_info['user_weights'] = dict(user_weights)
        
        if len(test_infos) == 0:
            test_info['user_n'] = dict(user_n)
        
        test_infos.append(test_info)
        joblib.dump(test_infos, data_dir / f"test_infos-{run}.data")
    
    
def see():
    device = torch.device("cpu") 
    dataset = Dataset("raw_data-1m.test", device)
    
    for _ in range(1000):
        idx = np.random.randint(len(dataset.raw_data['id']))
        display(dataset, idx, show_raw_text=True)
    
    
if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "test":
        test(sys.argv[2])
    elif sys.argv[1] == "split":
        split_dataset(sys.argv[2], float(sys.argv[3]))
    elif sys.argv[1] == "see":
        see()