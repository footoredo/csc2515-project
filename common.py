import numpy as np
from pathlib import Path
import time
from datetime import datetime, timedelta
import copy
import re
import csv


data_dir = Path("~/gobi2/data/project").expanduser()

class PriceChecker:
    
    def __init__(self, price_file="BTC-USD.csv"):
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
        
    def get_trend(self, d0, span=2, only_past=False, only_future=False):
        if only_future:
            d = d0
        elif only_past:
            d = d0 - timedelta(days=span - 1)
        else:
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
        return m / np.mean(ps) * 200
    
    def get_yesterday(self, d0):
        d1 = d0 - timedelta(days=1)
        p0 = self.get_price(d0)
        p1 = self.get_price(d1)
        
        if p0 is None or p1 is None:
            return None
        
        return (p0 - p1) / p1 * 100
    
    
def load_twitter_glove(dim):
    st = time.time()
    print(f"Loading twitter glove dim={dim}...")
    vecs = []
    words = []
    idx_map = dict()
    with open(str(data_dir / "glove" / f"glove.twitter.27B.{dim}d.txt")) as f:
        for i, line in enumerate(f):
            contents = line.split(' ')
            token = contents[0]
            # if token[0] == '<' and token[-1] == '>':
            #     print(token)
            vec = np.array(list(map(float, contents[1:])), dtype=np.float)
            # if vec.shape[0] != dim:
            #     print("!!!", line, len(contents))
            # if i == 0:
            #     print(vec)
            # print(i, vec.shape)
            idx_map[token] = i
            vecs.append(vec)
            words.append(token)
    # print(vecs[-1])
    # print("mean:", np.mean(vecs), "std:", np.std(vecs))
    # vecs.append(np.random.normal(loc=np.mean(vecs), scale=np.std(vecs), size=dim))
    vecs.append(np.zeros(dim, dtype=np.float))
    # print(vecs[-1])
    # print(vecs[-1].shape)
    words.append("<pad>")
    idx_map["<pad>"] = len(words) - 1
    print(f"Loading done. Duration: {time.time() - st:.2f}s.")
    return np.stack(vecs, 0), idx_map, words
    
    
def twitter_preprocess(text):
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<URL>", text)
    text = re.sub(r"[<>]", " ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"<3", "<HEART>", text)
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub(r"#(\S+)", r"<HASHTAG> \1", text)
    text = re.sub(r"([!?.]){2,}", r"\1 <REPEAT> ", text)
    text = re.sub(r"([!?.,])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z0-9!?.,<>]", " ", text)
    # print(text)
    text = re.sub(r"\b(\S*?)(\w)\2{2,}\b", r"\1\2 <ELONG>", text)
    text = re.sub(r"(?!>)\b[A-Z]{2,}\b(?!>)", lambda pat: pat.group(0).lower() + " <ALLCAPS>", text)
    # text = re.sub(r"(?!>)\b[a-zA-Z0-9]{2,}\b(?!>)", lambda pat: pat.group(0).lower(), text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    
    return text
    
    
if __name__ == "__main__":
    text = """"CHANGE IS COMING...GET READY!!! Boom, Another [CB] Jab, Nothing Can Stop This! Globalism at its end stage, [CB] push to make a one world govt. coming to an end. 

The People Are Taking the Lead, &amp; Leaders Will Have to Follow the Majority!!!

WWG1WGQ!!!

https://t.co/tAjFwxnWD4 One of the useful articles of Stefan; here is the guide, you can run a @LTOnetwork  node on Alibaba Cloud.

https://t.co/iJ9rlkaabt

#ltonetwork $lto
#Eth #xrpcommmunity #crypto #xlm
#xrp #blockchain #xrpcommmunity <fuck> #eos #xmr #trx #ltc #enjin  #ethereum #bitcoin"🐻🔫💥🐂🚀🙏 <fuck><fuck>
    """
    
    # text = "#bitcoin\"🐻🔫💥🐂🚀🙏 <fuck><fuck>"
    
    text = """CHANGE IS COMING...GET READY!!! Boom, Another [CB] Jab, Nothing Can Stop This! Globalism at its end stage, [CB] push to make a one world govt. coming to an end. 

The People Are Taking the Lead, &amp; Leaders Will Have to Follow the Majority!!!

WWG1WGQ!!!

https://t.co/tAjFwxnWD4"""
    
    text = twitter_preprocess(text)
    print(text)
    print(len(text.split()))