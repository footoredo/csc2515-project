import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import string
import copy
import csv
import time
import numpy as np
from collections import Counter
from collections import defaultdict

data_dir = Path("~/gobi2/data/project").expanduser()
TEXT_LIMIT = 60
VOCAB_SIZE = 100003
# 0 : None
# 1 : EOT
# 2 : number
# 3-100002 : word
# 100003-112188 : user >= 10
# 112189: user < 10
USER_SIZE = 12186 + 1
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 200


def reverse(o_text_idx, o_text_v, verbose=False):
    text_idx = np.zeros_like(o_text_idx)
    text_v = np.zeros_like(o_text_v)
    
    ending = None
    for i in range(TEXT_LIMIT):
        if o_text_idx[i] == 1:
            ending = i
            break
    
    if ending is not None:
        text_idx[TEXT_LIMIT - ending - 1] = 1
    else:
        ending = TEXT_LIMIT
    text_idx[TEXT_LIMIT - ending:] = o_text_idx[:ending]
    text_v[TEXT_LIMIT - ending:] = o_text_v[:ending]
    # for i in range(ending):
    #     text_idx[i + TEXT_LIMIT - ending] = o_text_idx[i]
    #     text_v[i + TEXT_LIMIT - ending] = o_text_v[i]
    
    if verbose:
        print(o_text_idx)
        print(ending)
        print(text_idx)
            
    return text_idx, text_v


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, extra_input_dim, hidden_dim, vocab_size, user_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.user_embeddings = nn.Embedding(user_size, 16)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # self.lstmw = nn.Linear(hidden_dim, hidden_dim)
        self.lstmln = nn.LayerNorm(hidden_dim)

        self.extraw = nn.Linear(16 + extra_input_dim, hidden_dim)
        self.extra2gate = nn.Linear(hidden_dim, 1)
        self.extraln = nn.LayerNorm(hidden_dim)
        self.hidden2scalar = nn.Linear(hidden_dim, 1)

    def forward(self, sentence, user_idx, extra_inpts):
        embeds = self.word_embeddings(sentence)
        # print("embeds:", embeds.shape)
        # print(embeds[0].detach().cpu().numpy())
        # print(embeds[1].detach().cpu().numpy())
        # print(embeds.shape, extra_inpts.shape)
        # inpt = torch.cat([embeds, extra_inpts], -1)
        inpt = embeds
        lstm_out, _ = self.lstm(inpt)
        lstm_out = lstm_out[:, -1, :]
        # print("lstm_out:", lstm_out.shape)
        # print(lstm_out[0].detach().cpu().numpy())
        # print(lstm_out[1].detach().cpu().numpy())
        # lstm_out = F.relu(self.lstmw(lstm_out))
        lstm_out = self.lstmln(lstm_out)
        
        # print(scalar_out[0].detach().cpu().numpy())
        # print(scalar_out[1].detach().cpu().numpy())

        user_embeds = self.user_embeddings(user_idx)
        extras = torch.cat([user_embeds, extra_inpts], -1)
        extras = F.relu(self.extraln(self.extraw(extras)))
        gate = torch.tanh(self.extra2gate(extras))
        # print(scalar_out.shape)
        # return torch.sigmoid(scalar_out)
        
        # out = torch.cat([lstm_out, extras], 1)
        out = lstm_out
        scalar_out = self.hidden2scalar(out) * gate
        
        return scalar_out


class PriceChecker:
    
    def __init__(self, price_file):
        path = data_dir / price_file
        self.data = dict()
        self.start_date = datetime(year=2016, month=1, day=1)
        with open(path) as price_data:
            reader = csv.DictReader(price_data)
            for row in reader:
                date = self.convert_date(datetime.fromisoformat(row['Date']))
                self.data[date] = copy.deepcopy(row)

    def convert_date(self, date):
        return (date - self.start_date).days
        
    def query(self, year, month, day):
        return self.query_from_datetime(datetime(year=year, month=month, day=day))

    def query_fromtimestamp(self, timestamp, verbose=False):
        # return self.data[self.convert_date(datetime.fromtimestamp(timestamp))]
        if verbose:
            print(timestamp, datetime.fromtimestamp(timestamp))
        return self.query_from_datetime(datetime.fromtimestamp(timestamp))

    def query_from_datetime(self, dt):
        try:
            return self.data[self.convert_date(datetime(year=dt.year, month=dt.month, day=dt.day))]
        except KeyError:
            return None
        
    def get_price(self, dt, key="Open"):
        data = self.query_from_datetime(dt)
        if data is None:
            return None
        return float(data[key])
        
    def get_trend(self, d0, span=2):
        d = d0 - timedelta(days=span // 2)
        ps = [self.get_price(d)]
        for _ in range(span - 1):
            d += timedelta(days=1)
            ps.append(self.get_price(d))
            
        # print(ps)
            
        for p in ps:
            if p is None:
                return None
            
        x = np.array(list(range(span)), dtype=np.float64)
        y = np.array(ps)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # print(ps, m)
        return m / np.mean(ps) * 100


class Dataset(torch.utils.data.Dataset):

    def __init__(self, text_file, price_file, device):
        text_data = joblib.load(data_dir / text_file)
        self.user_idx_dict = joblib.load(data_dir / "user_idx.obj")
        word_index = joblib.load(data_dir / "word_index-clean.obj")
        r_word_index = [None] * 100000
        for k, v in word_index.items():
            r_word_index[v] = k
        self.text_data = text_data
        self.device = device
        n = len(text_data['id'])
        price_checker = PriceChecker(price_file)
        self.text_idx_data = []
        self.text_v_data = []
        self.price_data = []
        self.user_idx_data = []
        self.rlr_data = []
        self.idx = []
        date_distribution = defaultdict(int)
        # print(text_data['text_idx'].max())
        # user_dict = defaultdict(int)
        print("loading dataset from", text_file, flush=True)
        st0 = time.time()
        st = time.time()
        for i in range(n):
            ts = text_data['timestamp'][i]
            d0 = datetime.fromtimestamp(ts)
            trend = price_checker.get_trend(d0, 5)
            if trend is None:
                continue
            # diff = (pp0 - pp1) / pp1 * 100
            self.price_data.append(trend)
            try:
                # user_idx = self.user_idx_dict[text_data['user'][i]] + 100004
                user_idx = self.user_idx_dict[text_data['user'][i]]
            except KeyError:
                # user_idx = 100003
                user_idx = 0
            # if abs(diff) > 15: 
            #     print(pp0, pp1, diff, text_data['user'][i], user_idx)
            #     for j in range(TEXT_LIMIT):
            #         idx = text_data['text_idx'][i][j]
            #         if idx == 0:
            #             word = "<NONE>"
            #         elif idx == 1:
            #             word = "<EOT>"
            #         elif idx == 2:
            #             word = "<NUM>"
            #         else:
            #             word = r_word_index[idx - 3]
            #         print(word, end=" ")
            #         if idx == 1:
            #             break
            #     print()
            # self.text_idx_data.append(np.concatenate([[user_idx], text_data['text_idx'][i]], -1))
            # self.text_v_data.append(np.concatenate([[0.], text_data['text_v'][i]], -1))
            # text_idx, text_v = reverse(text_data['text_idx'][i], text_data['text_v'][i])
            text_idx, text_v = text_data['text_idx'][i], text_data['text_v'][i]
            date = price_checker.convert_date(d0)
            self.text_idx_data.append(text_idx)
            self.text_v_data.append(text_v)
            self.user_idx_data.append(user_idx)
            self.rlr_data.append(np.array([d0.hour * 60 + d0.minute, text_data['replies'][i], text_data['likes'][i], text_data['retweets'][i]]))
            self.idx.append(i)
            date_distribution[date] += 1
            
            if i % 10000 == 0:
                t = time.time()
                print(f"#{i}, time: {t - st0:.2f}s, timedelta: {t - st:.2f}s", flush=True)
                st = t
            # user_dict[text_data['user'][i]] += 1
        print("loading dataset from", text_file, "done", flush=True)

        # print(len(user_dict))
        # uniq_users = sorted(user_dict.keys(), key=user_dict.get, reverse=True)
        # user_idx = dict()
        # for i, user in enumerate(uniq_users):
        #     if i < 10:
        #         print(i, user, user_dict[user])
        #     if user_dict[user] < 10:
        #         print("#user >= 10:", i)
        #         break
        #     user_idx[user] = i
        
        # joblib.dump(user_idx, data_dir / "user_idx.obj")

        self.n = len(self.price_data)
        self.text_idx_data = torch.tensor(np.array(self.text_idx_data)).to(self.device)
        self.text_v_data = torch.tensor(np.array(self.text_v_data), dtype=torch.float).unsqueeze(-1).to(self.device)
        self.price_data = torch.tensor(np.array(self.price_data), dtype=torch.float).unsqueeze(-1).to(self.device)
        self.user_idx_data = torch.tensor(np.array(self.user_idx_data)).to(self.device)
        self.rlr_data = torch.tensor(np.array(self.rlr_data), dtype=torch.float).to(self.device)
        self.idx = torch.tensor(self.idx).to(self.device)
        

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            self.text_idx_data[idx],
            self.text_v_data[idx],
            self.price_data[idx],
            torch.tensor(self.idx)
        )

    def get_data(self, key, idx):
        return self.text_data[key][idx]

    def reset(self):
        r = torch.randperm(self.n).to(self.device)
        self.text_idx_data = self.text_idx_data[r]
        self.text_v_data = self.text_v_data[r]
        self.price_data = self.price_data[r]
        self.user_idx_data = self.user_idx_data[r]
        self.rlr_data = self.rlr_data[r]
        self.idx = self.idx[r]
        self.cur = 0

    def iterate_batch(self, batch_size):
        self.reset()

        while self.cur < self.n:
            r = min(self.cur + batch_size, self.n)
            yield self.text_idx_data[self.cur: r], \
                self.text_v_data[self.cur: r], \
                self.user_idx_data[self.cur: r], \
                self.rlr_data[self.cur: r], \
                self.price_data[self.cur: r], \
                self.idx[self.cur: r]
            self.cur += batch_size


def main():
    print(torch.cuda.is_available())
    # device = torch.device("cpu") 
    device = torch.device("cuda:0") 
    dataset = Dataset("cleaned_data-5m.obj", "BTC-USD.csv", device)
    # return
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = Dataset("cleaned_data-test-1m.obj", "BTC-USD.csv", device)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, VOCAB_SIZE, USER_SIZE)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    infos = []
    train_infos = []
    test_infos = []

    # criterion = nn.BCELoss()

    print("start training")

    for epoch in range(10):
        st = time.time()
        # for batch, (text_idx, text_v, price, idx) in enumerate(dataloader):
        epoch_loss = 0
        epoch_acc = 0
        epoch_n = 0
        for batch, (text_idx, text_v, user_idx, rlr, price, idx) in enumerate(dataset.iterate_batch(BATCH_SIZE)):
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

            price_pred = model(text_idx, user_idx, rlr)
            if batch == 0:
                if epoch == 0:
                    print("text_idx", text_idx.shape)
                    print("user_idx", user_idx.shape)
                    print("rlr", rlr.shape)
                    print("price", price.shape)
                    print(text_idx[0].cpu().numpy())
                    print(text_idx[1].cpu().numpy())
                print(price_pred[:20, 0].detach().cpu().numpy())
                print(price[:20, 0].cpu().numpy())
            loss = (0.5 * (price_pred - price) ** 2).mean()
            # loss = criterion(price_pred, _price)
            acc = ((price_pred > 0).float() - _price).abs().mean()

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
        train_info = { 'epoch': epoch, 'loss': epoch_loss / epoch_n, 'acc': epoch_acc / epoch_n }
        train_infos.append(train_info)
        print("train:", train_info)
        torch.save(model.state_dict(), str(data_dir / f"model-b-single-new-{epoch}.pkl"))
        sum_loss = 0.
        sum_acc = 0.
        sum_n = 0
        for batch, (text_idx, text_v, user_idx, rlr, price, idx) in enumerate(test_dataset.iterate_batch(BATCH_SIZE)):
            text_idx = text_idx.to(device)
            text_v = text_v.to(device)
            price = price.to(device)
            _price = (price > 0).float()
            price_pred = model(text_idx, user_idx, rlr)
            # loss = criterion(price_pred, _price)
            loss = (0.5 * (price_pred - price) ** 2).mean()
            acc = ((price_pred > 0).float() - _price).abs().mean()
            # loss = (0.5 * (price_pred - _price) ** 2).mean()
            sum_loss += loss.item() * price.shape[0]
            sum_acc += acc.item() * price.shape[0]
            sum_n += price.shape[0]
        test_info = { 'epoch': epoch, 'loss': sum_loss / sum_n, 'acc': sum_acc / sum_n }
        print("test:", test_info)
        test_infos.append(test_info)

    joblib.dump(infos, data_dir / "infos-new.data")
    joblib.dump(test_infos, data_dir / "test_infos-new.data")
    joblib.dump(train_infos, data_dir / "train_infos-new.data")
    torch.save(model.state_dict(), str(data_dir / "model-new.pkl"))


def test():
    # dataset = Dataset("cleaned_data-test-mini.obj", "BTC-USD.csv")
    # device = torch.device("cpu") 
    device = torch.device("cuda:0") 
    dataset = Dataset("cleaned_data-test-1m.obj", "BTC-USD.csv", device)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, VOCAB_SIZE, USER_SIZE)
    model.to(device)
    
    top20_infos = []
    
    for epoch in range(0, 90, 1):
        print(f"--- Epoch {epoch} ---", flush=True)
        
        model.load_state_dict(torch.load(str(data_dir / f"model-b-single-{epoch}.pkl")))
    
        sum_loss = 0.
        sum_acc = 0.
        sum_n = 0
        user_loss = defaultdict(float)
        user_acc = defaultdict(float)
        user_n = defaultdict(int)
        for batch, (text_idx, text_v, user_idx, rlr, price, idx) in enumerate(dataset.iterate_batch(BATCH_SIZE)):
            price_pred = model(text_idx, user_idx, rlr)
            loss = (0.5 * (price_pred - price) ** 2)
            acc = (torch.sign(price) == torch.sign(price_pred)).float()
            sum_loss += loss.mean().item() * price.shape[0]
            sum_acc += acc.mean().item() * price.shape[0]
            sum_n += price.shape[0]

            for i in range(price.shape[0]):
                user = dataset.get_data('user', idx[i].item())
                user_acc[user] += acc[i].item()
                user_loss[user] += loss[i].item()
                user_n[user] += 1

        for user in user_n.keys():
            user_acc[user] /= user_n[user]
            user_loss[user] /= user_n[user]
            
        info = {
            "avg_loss": sum_loss / sum_n,
            "avg_acc": sum_acc / sum_n
        }

        print(sum_loss / sum_n, flush=True)
        print(sum_acc / sum_n, flush=True)
        
        sum_acc = 0.

        print("\nTop 20 acc:")
        users = sorted(user_acc.keys(), key=user_acc.get, reverse=True)
        cnt = 0
        user_data = []
        for user in users:
            if user_n[user] >= 200:
                user_data.append((user, user_acc[user], user_loss[user], user_n[user]))
                cnt += 1
                if cnt <= 20:
                    print(f"#{cnt}", user, user_acc[user], user_n[user])
                    sum_acc += user_acc[user]
                    
        print("Top 20 acc avg acc:", sum_acc / 20)
        
        sum_loss = 0.
        
        print("\nTop 20 loss:")
        users = sorted(user_loss.keys(), key=user_loss.get)
        cnt = 0
        for user in users:
            if user_n[user] >= 200:
                cnt += 1
                if cnt <= 20:
                    print(f"#{cnt}", user, user_loss[user], user_n[user])
                    sum_loss += user_loss[user]
                    
        print("Top 20 loss avg loss:", sum_loss / 20)
        
        info["top20_acc"] = sum_acc / 20
        info["top20_loss"] = sum_loss / 20
        
        top20_infos.append(info)

        print("# >=200 accounts:", cnt, flush=True)

        joblib.dump(user_data, data_dir / f"user_test_data-{epoch}.obj")
    
    joblib.dump(top20_infos, data_dir / "top20_infos.obj")


def human_test():
    word_index = joblib.load(data_dir / "word_index-clean.obj")
    from a import index_word
    model = LSTMModel(EMBEDDING_DIM, 1, HIDDEN_DIM, VOCAB_SIZE)
    model.load_state_dict(torch.load(str(data_dir / "model.pkl")))

    print("Ready!")

    while True:
        v = input()

        cleaned_words_idx = np.zeros(TEXT_LIMIT, dtype=int)
        cleaned_words_v = np.zeros((TEXT_LIMIT, 1), dtype=float)
        # print(v.split())
        words = v.split()
        for word_i, word in enumerate(words):
            if word_i >= TEXT_LIMIT:
                break
            # print(word, index_word(word, self.word_index))
            idx, v = index_word(word, word_index)
            cleaned_words_idx[word_i] = idx
            cleaned_words_v[word_i] = v
        if len(words) < TEXT_LIMIT:
            cleaned_words_idx[len(words)] = 1
        
        price_pred = model(torch.tensor(cleaned_words_idx).unsqueeze(0), torch.zeros((TEXT_LIMIT, 1), dtype=torch.float).unsqueeze(0))

        print(price_pred.item())



if __name__ == "__main__":
    main()
