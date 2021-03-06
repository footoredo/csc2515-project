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

# data_dir = Path("~/ssd004/data/project").expanduser()
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
BATCH_SIZE = 50


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
    for i in range(ending):
        text_idx[i + TEXT_LIMIT - ending] = o_text_idx[i]
        text_v[i + TEXT_LIMIT - ending] = o_text_v[i]
    
    if verbose:
        print(o_text_idx)
        print(ending)
        print(text_idx)
            
    return text_idx, text_v


class LSTMModelOld(nn.Module):

    def __init__(self, embedding_dim, extra_input_dim, hidden_dim, vocab_size, user_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.user_embeddings = nn.Embedding(user_size, 16)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.extraw = nn.Linear(16 + extra_input_dim, hidden_dim)
        self.extra2gate = nn.Linear(hidden_dim, 1)
        self.hiddenw = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2scalar = nn.Linear(hidden_dim, 1)

    def forward(self, sentence, user_idx, extra_inpts):
        embeds = self.word_embeddings(sentence)
        # print(embeds.shape, extra_inpts.shape)
        # inpt = torch.cat([embeds, extra_inpts], -1)
        inpt = embeds
        lstm_out, _ = self.lstm(inpt)
        scalar_out = F.relu(self.hiddenw(lstm_out[:, -1, :]))
        scalar_out = self.hidden2scalar(scalar_out)

        user_embeds = self.user_embeddings(user_idx)
        extras = torch.cat([user_embeds, extra_inpts], -1)
        extras = F.relu(self.extraw(extras))
        gate = torch.tanh(self.extra2gate(extras))
        # print(scalar_out.shape)
        # return torch.sigmoid(scalar_out)
        return scalar_out * gate


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, extra_input_dim, hidden_dim, vocab_size, user_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.user_embeddings = nn.Embedding(user_size, 16)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.extraw = nn.Linear(16 + extra_input_dim + hidden_dim, hidden_dim)
        self.extra2gate = nn.Linear(hidden_dim, 1)
        self.hiddenw1 = nn.Linear(hidden_dim, hidden_dim)
        self.hiddenw2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2scalar = nn.Linear(hidden_dim, 1)

    def forward(self, sentence, user_idx, extra_inpts, mask, debug=False):
        # sentence: [batch, num_tweets, length]
        # mask: [batch, num_tweets]
        batch_size, num_tweets, length = sentence.shape
        embeds = self.word_embeddings(sentence)  # [batch, num_tweets, length, embed]
        # print(embeds.shape, extra_inpts.shape)
        # inpt = torch.cat([embeds, extra_inpts], -1)
        inpt = embeds.view(batch_size * num_tweets, length, self.embedding_dim)
        lstm_out, _ = self.lstm(inpt)
        lstm_out = lstm_out[:, -1].view(batch_size, num_tweets, self.hidden_dim)
        lstm_out = F.relu(self.hiddenw1(lstm_out))
        out = F.relu(self.hiddenw2(lstm_out))

        user_embeds = self.user_embeddings(user_idx)
        extras = torch.cat([user_embeds, extra_inpts, lstm_out], -1)
        extras = F.relu(self.extraw(extras))
        extras_logits = self.extra2gate(extras)[:, :, 0]  # [batch, num_tweets]
        extras_logits += -1e5 * (1 - mask)
        gate = F.softmax(extras_logits, 1).unsqueeze(-1)

        out = (out * gate).sum(1)  # [batch, hidden]

        scalar_out = self.hidden2scalar(out)
        scalar_out = torch.sigmoid(scalar_out)
        # print(scalar_out.shape)
        # return torch.sigmoid(scalar_out)
        if debug:
            return scalar_out, gate
        else:
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
        dt.hour = 0
        dt.minute = 0
        dt.second = 0
        try:
            return self.data[self.convert_date(dt)]
        except KeyError:
            return None
        
    def get_trend(self, d0, span=2):
        d = d0 - timedelta(days=span // 2)
        ps = [self.query_from_datetime(d)]
        for _ in range(span - 1):
            d += timedelta(days=1)
            ps.append(self.query_from_datetime(d))
            
        if any(ps == None):
            return None
            
        x = np.array(list(range(span)))
        y = np.array(ps)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        # print(ps, m)
        return m / np.mean(ps) * 100


class Dataset(torch.utils.data.Dataset):

    def __init__(self, text_file, price_file, device, filter_dates=False):
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
        self.text_idx_data = defaultdict(list)
        self.text_v_data = defaultdict(list)
        self.price_data = defaultdict(list)
        self.user_idx_data = defaultdict(list)
        self.rlr_data = defaultdict(list)
        self.idx = defaultdict(list)
        date_distribution = defaultdict(int)
        # print(text_data['text_idx'].max())
        # user_dict = defaultdict(int)
        for i in range(n):
            ts = text_data['timestamp'][i]
            d0 = datetime.fromtimestamp(ts)
            trend = price_checker.get_trend(d0, 5)
            if trend is None:
                continue
            # p0 = price_checker.query_from_datetime(d0)
            # # should be day not second!!
            # d1 = d0 - timedelta(days=1)
            # p1 = price_checker.query_from_datetime(d1)
            # if p0 is None or p1 is None:
            #     continue
            # pp0 = float(p0['Open'])
            # pp1 = float(p1['Open'])
            # # diff = (pp1 - pp0) / pp0 * 100
            # diff = (pp0 - pp1) / pp1 * 100
            # assert pp0 > 0 and pp1 > 0
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
            text_idx, text_v = reverse(text_data['text_idx'][i], text_data['text_v'][i])
            date = price_checker.convert_date(d0)
            self.price_data[date].append(trend)
            self.text_idx_data[date].append(text_idx)
            self.text_v_data[date].append(text_v)
            self.user_idx_data[date].append(user_idx)
            self.rlr_data[date].append(np.array([d0.hour * 60 + d0.minute, text_data['replies'][i], text_data['likes'][i], text_data['retweets'][i]]))
            self.idx[date].append(i)
            date_distribution[date] += 1
            # user_dict[text_data['user'][i]] += 1

        valid_dates = []
        sum_n = 0
        for date, n in date_distribution.items():
            sum_n += n
            if not filter_dates or n >= filter_dates:
                valid_dates.append(date)
        print(f"num dates: {len(date_distribution.keys())}, sum_n: {sum_n}")

        valid_dates = sorted(valid_dates)

        for date in valid_dates:
            self.text_idx_data[date] = torch.tensor(np.array(self.text_idx_data[date])).to(self.device)
            self.text_v_data[date] = torch.tensor(np.array(self.text_v_data[date]), dtype=torch.float).unsqueeze(-1).to(self.device)
            # print(date, np.var(self.price_data[date]))
            self.price_data[date] = torch.tensor(np.array(self.price_data[date]), dtype=torch.float).unsqueeze(-1).to(self.device)
            self.user_idx_data[date] = torch.tensor(np.array(self.user_idx_data[date])).to(self.device)
            self.rlr_data[date] = torch.tensor(np.array(self.rlr_data[date]), dtype=torch.float).to(self.device)
            self.idx[date] = torch.tensor(self.idx[date]).to(self.device)

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

        self.n = len(valid_dates)
        self.valid_dates = valid_dates
        self.date_distributions = date_distribution
        # self.text_idx_data = torch.tensor(np.array(self.text_idx_data)).to(self.device)
        # self.text_v_data = torch.tensor(np.array(self.text_v_data), dtype=torch.float).unsqueeze(-1).to(self.device)
        # self.price_data = torch.tensor(np.array(self.price_data), dtype=torch.float).unsqueeze(-1).to(self.device)
        # self.user_idx_data = torch.tensor(np.array(self.user_idx_data)).to(self.device)
        # self.rlr_data = torch.tensor(np.array(self.rlr_data), dtype=torch.float).to(self.device)
        # self.idx = torch.tensor(self.idx).to(self.device)
        

    def __len__(self):
        return len(self.price_data)

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

    def build_dateset(self, num_tweets):
        dates = copy.deepcopy(self.valid_dates)
        np.random.shuffle(dates)
        text_idx_data = []
        text_v_data = []
        price_data = []
        user_idx_data = []
        rlr_data = []
        idx = []
        mask = []
        for date in dates:
            n = self.date_distributions[date]
            indices = list(range(n))
            np.random.shuffle(indices)
            if n < num_tweets:
                for _ in range(num_tweets - n):
                    indices.append(0)
            else:
                indices = indices[:num_tweets]
            text_idx_data.append(self.text_idx_data[date][indices])
            text_v_data.append(self.text_v_data[date][indices])
            user_idx_data.append(self.user_idx_data[date][indices])
            rlr_data.append(self.rlr_data[date][indices])
            idx.append(self.idx[date][indices])

            _mask = [1] * min(n, num_tweets) + [0] * max(0, num_tweets - n)
            mask.append(torch.tensor(_mask).to(self.device))
            price_data.append(self.price_data[date][0])

        text_idx_data = torch.stack(text_idx_data, 0)
        text_v_data = torch.stack(text_v_data, 0)
        user_idx_data = torch.stack(user_idx_data, 0)
        rlr_data = torch.stack(rlr_data, 0)
        idx = torch.stack(idx, 0)
        mask = torch.stack(mask, 0)
        price_data = torch.stack(price_data, 0)

        return text_idx_data, text_v_data, user_idx_data, rlr_data, mask, price_data, idx


    def iterate_batch(self, batch_size, num_tweets):
        text_idx_data, text_v_data, user_idx_data, rlr_data, mask, price_data, idx = self.build_dateset(num_tweets)
        cur = 0

        while cur < self.n:
            r = min(cur + batch_size, self.n)
            yield text_idx_data[cur: r], \
                text_v_data[cur: r], \
                user_idx_data[cur: r], \
                rlr_data[cur: r], \
                mask[cur: r], \
                price_data[cur: r], \
                idx[cur: r]
            cur += batch_size


def main():
    print(torch.cuda.is_available())
    # device = torch.device("cpu") 
    device = torch.device("cuda:0") 
    dataset = Dataset("cleaned_data-1m.obj", "BTC-USD.csv", device, filter_dates=1)
    # return
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = Dataset("cleaned_data-test-mini.obj", "BTC-USD.csv", device)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, VOCAB_SIZE, USER_SIZE)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    infos = []
    train_infos = []
    test_infos = []

    criterion = nn.BCELoss()

    print("start training")

    for epoch in range(100):
        st = time.time()
        # for batch, (text_idx, text_v, price, idx) in enumerate(dataloader):
        epoch_loss = 0
        epoch_acc = 0
        epoch_n = 0
        for batch, (text_idx, text_v, user_idx, rlr, mask, price, idx) in enumerate(dataset.iterate_batch(BATCH_SIZE, 1)):
            if False and epoch == 0 and batch == 0:
                print(text_idx.shape)
                print(text_idx[0].cpu().numpy())
                print(user_idx.shape)
                print(user_idx[0].cpu().numpy())
                print(rlr.shape)
                print(rlr[0].cpu().numpy())
                print(mask.shape)
                print(mask[0].cpu().numpy())
                print(price.shape)
                print(price[0].cpu().numpy())
                return
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

            price_pred = model(text_idx, user_idx, rlr, mask)
            # loss = (0.5 * (price_pred - price) ** 2).mean()
            loss = criterion(price_pred, _price)
            # acc = ((price_pred > 0).float() - _price).abs().mean()
            acc = ((price_pred > 0.5).float() - _price).abs().mean()

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
        torch.save(model.state_dict(), str(data_dir / f"model-b-{epoch}.pkl"))
        sum_loss = 0.
        sum_acc = 0.
        sum_n = 0
        for batch, (text_idx, text_v, user_idx, rlr, mask, price, idx) in enumerate(test_dataset.iterate_batch(BATCH_SIZE, 1)):
            # text_idx = text_idx.to(device)
            # text_v = text_v.to(device)
            # price = price.to(device)
            _price = (price > 0).float()
            price_pred = model(text_idx, user_idx, rlr, mask)
            loss = criterion(price_pred, _price)
            # loss = (0.5 * (price_pred - price) ** 2).mean()
            # acc = ((price_pred > 0).float() - _price).abs().mean()
            acc = ((price_pred > 0.5).float() - _price).abs().mean()
            # loss = (0.5 * (price_pred - _price) ** 2).mean()
            sum_loss += loss.item() * price.shape[0]
            sum_acc += acc.item() * price.shape[0]
            sum_n += price.shape[0]
        test_info = { 'epoch': epoch, 'loss': sum_loss / sum_n, 'acc': sum_acc / sum_n }
        print("test:", test_info)
        test_infos.append(test_info)

    joblib.dump(infos, data_dir / "infos.data")
    joblib.dump(test_infos, data_dir / "test_infos.data")
    torch.save(model.state_dict(), str(data_dir / "model.pkl"))


def test():
    device = torch.device("cpu") 
    dataset = Dataset("cleaned_data-test-mini.obj", "BTC-USD.csv", device)
    # dataset = Dataset("cleaned_data-mini.obj", "BTC-USD.csv", device)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    model = LSTMModel(EMBEDDING_DIM, 4, HIDDEN_DIM, VOCAB_SIZE, USER_SIZE)
    model.to(device)
    model.load_state_dict(torch.load(str(data_dir / "model-1.pkl")))
    sum_loss = 0.
    sum_acc = 0.
    sum_n = 0
    user_gate = defaultdict(float)
    user_n = defaultdict(int)
    for batch, (text_idx, text_v, user_idx, rlr, mask, price, idx) in enumerate(dataset.iterate_batch(1, 30)):
        price_pred, gate = model(text_idx, user_idx, rlr, mask, debug=True)
        print(price[0].item(), price_pred[0].item(), mask[0].numpy())
        loss = (0.5 * (price_pred - price) ** 2)
        acc = (torch.sign(price) == torch.sign(price_pred)).float()
        sum_loss += loss.mean().item() * price.shape[0]
        sum_acc += acc.mean().item() * price.shape[0]
        sum_n += price.shape[0]

        for i in range(user_idx.shape[0]):
            for j in range(user_idx.shape[1]):
                if mask[i, j] > 0.5:
                    user = user_idx[i, j]
                    user_gate[user] += gate[i, j].item()
                    user_n[user] += 1
        # break

    for user in user_gate.keys():
        user_gate[user] /= user_n[user]

    users = sorted(user_gate.keys(), key=user_gate.get, reverse=True)
    for user in users:
        if user_n[user] > 100:
            print(user, user_gate[user], user_n[user])


    print(sum_loss / sum_n)
    print(sum_acc / sum_n)


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
