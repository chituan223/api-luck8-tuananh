from flask import Flask, jsonify
import requests
import time
import threading
import math
import json
import os
from collections import deque, Counter
from datetime import datetime

app = Flask(__name__)

# ================= CONFIG =================
MIN_PHIEN = 15
MAX_HISTORY = 20
DATA_FILE = "luckywin_ai_data.json"
WEIGHTS_FILE = "luckywin_ai_weights.json"

API_LIST = [
    "https://luck8bot.com/api/GetNewLottery/TaixiuMd5?id=",
    "http://luck8bot.com/api/GetNewLottery/TaixiuMd5?id="
]

# ================= DATA =================
history_tx = deque(maxlen=MAX_HISTORY)
history_pt = deque(maxlen=MAX_HISTORY)
history_id = deque(maxlen=MAX_HISTORY)

lich_su = []
stats = {"tong": 0, "dung": 0, "sai": 0, "cd": 0, "cs": 0, "max_cd": 0, "max_cs": 0}

last_result = {}
_prev_pred = None

# ================= UTILS =================
def encode(tx_list):
    return "".join(["T" if x == "Tài" else "X" for x in tx_list])

def decode(c):
    return "Tài" if c == "T" else "Xỉu"

def save_data():
    try:
        data = {
            "history": list(history_tx),
            "points": list(history_pt),
            "ids": list(history_id),
            "stats": stats,
            "lich_su": lich_su[-100:]
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

def load_data():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                for h, p, i in zip(data.get("history", [])[-100:],
                                   data.get("points", [])[-100:],
                                   data.get("ids", [])[-100:]):
                    history_tx.append(h)
                    history_pt.append(p)
                    history_id.append(i)
    except:
        pass

# ================= 20 AI MODELS - TỐI ƯU NHẤT =================

class MarkovChain:
    def predict(self, tx_list, order=3):
        if len(tx_list) < order + 5:
            return None, 0, "Need data"
        s = encode(tx_list)
        trans = {}
        for i in range(len(s) - order):
            state = s[i:i+order]
            trans.setdefault(state, Counter())[s[i+order]] += 1
        cur = s[-order:]
        if cur not in trans or sum(trans[cur].values()) < 3:
            return None, 0, "No state"
        counts = trans[cur]
        total = sum(counts.values())
        p_t = counts.get("T", 0) / total
        conf = max(p_t, 1-p_t)
        if conf < 0.55:
            return None, 0, "Weak signal"
        return ("Tài" if p_t > 0.5 else "Xỉu"), conf * 100, f"Markov{order}"

class NGramModel:
    def predict(self, tx_list, max_n=5):
        if len(tx_list) < 12:
            return None, 0, "Need data"
        s = encode(tx_list)
        votes = Counter()
        confs = []
        for n in range(2, min(max_n+1, len(s)//2)):
            if len(s) < n + 3:
                continue
            grams = Counter()
            for i in range(len(s)-n):
                grams[(s[i:i+n], s[i+n])] += 1
            cur = s[-n:]
            t_score = sum(v for (g,c),v in grams.items() if g==cur and c=="T")
            x_score = sum(v for (g,c),v in grams.items() if g==cur and c=="X")
            total = t_score + x_score
            if total >= 3:
                if t_score > x_score:
                    votes["Tài"] += n * (t_score/total)
                    confs.append(t_score/total)
                elif x_score > t_score:
                    votes["Xỉu"] += n * (x_score/total)
                    confs.append(x_score/total)
        if not votes or not confs:
            return None, 0, "No ngram"
        pred = votes.most_common(1)[0][0]
        avg_conf = sum(confs)/len(confs)
        if avg_conf < 0.55:
            return None, 0, "Weak ngram"
        return pred, avg_conf * 100, "N-Gram"

class PatternModel:
    def predict(self, tx_list):
        if len(tx_list) < 5:
            return None, 0, "Need data"
        s = encode(tx_list)
        n = len(s)
        
        streak = 0
        last = s[-1]
        for c in reversed(s):
            if c == last: streak += 1
            else: break
        
        # Bệt mạnh -> đảo
        if streak >= 4:
            opp = "X" if last == "T" else "T"
            return decode(opp), min(55 + streak * 8, 85), f"Bệt{streak}"
        if streak == 3:
            opp = "X" if last == "T" else "T"
            return decode(opp), 65, f"Bệt3"
        
        # 1-1
        if n >= 5:
            recent = s[-5:]
            if all(recent[i] != recent[i+1] for i in range(len(recent)-1)):
                return decode("X" if recent[-1] == "T" else "T"), 70, "1-1"
        
        # 2-2
        if n >= 6:
            recent = s[-6:]
            if recent[0]==recent[1] and recent[2]==recent[3] and recent[4]==recent[5]:
                if recent[0]!=recent[2] and recent[2]!=recent[4]:
                    return decode(recent[4]), 72, "2-2"
        
        # Cân bằng
        if n >= 20:
            recent = s[-20:]
            t_pct = recent.count("T") / 20
            if t_pct >= 0.75:
                return "Xỉu", 70, f"Tài{t_pct:.0%}"
            elif t_pct <= 0.25:
                return "Tài", 70, f"Xỉu{1-t_pct:.0%}"
        
        return None, 0, "No pattern"

class StreakModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0, "Need data"
        s = encode(tx_list)
        streaks = []
        cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]: cur += 1
            else:
                streaks.append(cur)
                cur = 1
        streaks.append(cur)
        avg = sum(streaks) / len(streaks)
        current = streaks[-1]
        
        if current >= avg + 1.5:
            opp = "Xỉu" if s[-1] == "T" else "Tài"
            return opp, min(55 + (current-avg)*12, 80), f"Streak{current}>avg{avg:.1f}"
        return None, 0, "Streak TB"

class ReversalModel:
    def predict(self, tx_list):
        if len(tx_list) < 12:
            return None, 0, "Need data"
        s = encode(tx_list)[-12:]
        rev = sum(1 for i in range(1, len(s)) if s[i] != s[i-1])
        rate = rev / (len(s)-1)
        
        if rate >= 0.65:
            return decode("X" if s[-1] == "T" else "T"), 68, f"Đảo{rate:.0%}"
        elif rate <= 0.25:
            return decode(s[-1]), 68, f"ÍtĐảo{rate:.0%}"
        return None, 0, f"ĐảoTB"

class CycleModel:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0, "Need data"
        s = encode(tx_list)
        best_cycle = None
        best_score = 0
        
        for c_len in range(2, min(10, len(s)//3)):
            matches = sum(1 for i in range(c_len, min(len(s), c_len*10)) if s[i] == s[i-c_len])
            score = matches / min(len(s)-c_len, c_len*9)
            if score > best_score and score > 0.6:
                best_score = score
                best_cycle = c_len
        
        if best_cycle and best_score > 0.65:
            return decode(s[-best_cycle]), min(best_score*100, 85), f"ChuKỳ{best_cycle}"
        return None, 0, "No cycle"

class MomentumModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0, "Need data"
        s = encode(tx_list)
        signals = []
        for w in [5, 10, 15]:
            if len(s) >= w:
                t_pct = s[-w:].count("T") / w
                if t_pct >= 0.7:
                    signals.append(("Xỉu", t_pct))
                elif t_pct <= 0.3:
                    signals.append(("Tài", t_pct))
        
        if len(signals) >= 2:
            preds = [x[0] for x in signals]
            if len(set(preds)) == 1:
                conf = sum(x[1] for x in signals) / len(signals)
                return preds[0], min(conf*100, 80), "Momentum"
        return None, 0, "Mom TB"

class TrendModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0, "Need data"
        s = encode(tx_list)
        third = len(s) // 3
        t1 = s[:third].count("T") / third if third else 0.5
        t2 = s[third:2*third].count("T") / third if third else 0.5
        t3 = s[2*third:].count("T") / (len(s)-2*third) if len(s)-2*third else 0.5
        
        if t3 > t2 > t1 and t3 - t1 > 0.2:
            return "Tài", 75, "TrendUp"
        elif t3 < t2 < t1 and t1 - t3 > 0.2:
            return "Xỉu", 75, "TrendDown"
        return None, 0, "TrendFlat"

class MAModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0, "Need data"
        v = [1 if x == "Tài" else 0 for x in tx_list]
        ma5 = sum(v[-5:]) / 5
        ma10 = sum(v[-10:]) / 10
        ma20 = sum(v[-20:]) / 20
        
        if ma5 > ma10 > ma20 and ma5 > 0.55:
            return "Tài", 72, "MA Bull"
        elif ma5 < ma10 < ma20 and ma5 < 0.45:
            return "Xỉu", 72, "MA Bear"
        return None, 0, f"MA{ma5:.2f}"

class BayesianModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0, "Need data"
        s = encode(tx_list)
        prior_t = s.count("T") / len(s)
        
        # Likelihood từ 3 phiên gần nhất
        recent3 = s[-3:]
        t3 = recent3.count("T")
        
        if t3 == 3:  # Bệt 3 Tài -> Xỉu
            return "Xỉu", 65, "BayesBệt3T"
        elif t3 == 0:  # Bệt 3 Xỉu -> Tài
            return "Tài", 65, "BayesBệt3X"
        
        return None, 0, "Bayes TB"

class EntropyModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0, "Need data"
        s = encode(tx_list)[-15:]
        t = s.count("T")
        x = len(s) - t
        if t == 0 or x == 0:
            return decode("X" if s[-1] == "T" else "T"), 70, "EntropyEdge"
        
        p_t = t / len(s)
        p_x = 1 - p_t
        entropy = -(p_t*math.log2(p_t) + p_x*math.log2(p_x))
        
        if entropy < 0.6:
            return decode(s[-1]), 70, f"EntropyThấp"
        return None, 0, f"Entropy{entropy:.2f}"

class MarkovHighOrder:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0, "Need data"
        s = encode(tx_list)
        best_pred = None
        best_conf = 0
        
        for order in [2, 3, 4]:
            if len(s) < order + 5:
                continue
            trans = {}
            for i in range(len(s)-order):
                state = s[i:i+order]
                trans.setdefault(state, Counter())[s[i+order]] += 1
            cur = s[-order:]
            if cur not in trans:
                continue
            counts = trans[cur]
            total = sum(counts.values())
            if total < 3:
                continue
            p_t = counts.get("T", 0) / total
            conf = max(p_t, 1-p_t)
            if conf > best_conf and conf >= 0.6:
                best_conf = conf
                best_pred = "Tài" if p_t > 0.5 else "Xỉu"
        
        if best_pred:
            return best_pred, best_conf * 100, "MarkovHO"
        return None, 0, "MarkovHO TB"

class RegressionModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0, "Need data"
        v = [1 if x == "Tài" else 0 for x in tx_list[-15:]]
        n = len(v)
        # Đếm sự thay đổi gần đây
        changes = sum(1 for i in range(1, n) if v[i] != v[i-1])
        if changes <= 2:
            return decode("X" if encode(tx_list)[-1] == "T" else "T"), 65, "RegFlat"
        return None, 0, "RegTB"

class FreqAdaptiveModel:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0, "Need data"
        s = encode(tx_list)
        
        # Kiểm tra 3 cửa sổ đồng thuận
        signals = []
        for w in [10, 15, 25]:
            if len(s) < w:
                continue
            t_pct = s[-w:].count("T") / w
            if t_pct >= 0.72:
                signals.append("Xỉu")
            elif t_pct <= 0.28:
                signals.append("Tài")
        
        if len(signals) >= 2 and len(set(signals)) == 1:
            return signals[0], 75, f"Freq{len(signals)}W"
        return None, 0, "Freq TB"

class DeepPatternModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0, "Need data"
        s = encode(tx_list)
        
        # Tìm pattern lặp sau khoảng cách ngắn
        for length in range(3, 6):
            if len(s) < length * 3:
                continue
            sub = s[-length:]
            # Tìm lần xuất hiện trước
            for i in range(len(s)-length*2, -1, -1):
                if s[i:i+length] == sub:
                    if i+length < len(s):
                        next_char = s[i+length]
                        return decode(next_char), 70, f"DeepLặp{length}"
                    break
        return None, 0, "Deep TB"

class FibonacciModel:
    def predict(self, tx_list):
        if len(tx_list) < 8:
            return None, 0, "Need data"
        s = encode(tx_list)
        streak = 0
        last = s[-1]
        for c in reversed(s):
            if c == last: streak += 1
            else: break
        
        fib = [2, 3, 5, 8]
        if streak in fib:
            opp = "X" if last == "T" else "T"
            return decode(opp), 68, f"Fib{streak}"
        return None, 0, "Fib TB"

class GoldenRatioModel:
    def predict(self, tx_list):
        if len(tx_list) < 30:
            return None, 0, "Need data"
        s = encode(tx_list)
        phi = int(len(s) / 1.618)
        recent_phi = s[-phi:].count("T") / phi if phi else 0.5
        if recent_phi > 0.65:
            return "Xỉu", 65, "PhiTài"
        elif recent_phi < 0.35:
            return "Tài", 65, "PhiXỉu"
        return None, 0, "Phi TB"

class VolatilityModel:
    def predict(self, tx_list):
        if len(tx_list) < 12:
            return None, 0, "Need data"
        s = encode(tx_list)[-12:]
        changes = sum(1 for i in range(1, len(s)) if s[i] != s[i-1])
        vol = changes / (len(s)-1)
        
        if vol > 0.6:
            return decode("X" if s[-1] == "T" else "T"), 65, "VolCao"
        elif vol < 0.15:
            return decode(s[-1]), 65, "VolThấp"
        return None, 0, f"Vol{vol:.0%}"

class SupportResistanceModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0, "Need data"
        s = encode(tx_list)
        streaks = []
        cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]: cur += 1
            else:
                streaks.append(cur)
                cur = 1
        streaks.append(cur)
        
        if len(streaks) >= 2 and streaks[-2] >= 4:
            return decode(s[-1]), 70, f"SRBreak{streaks[-2]}"
        return None, 0, "SR TB"

class TimeSeriesModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0, "Need data"
        v = [1 if x == "Tài" else 0 for x in tx_list[-15:]]
        # Weighted recent
        weights = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
        weighted = sum(a*b for a,b in zip(v, weights[-len(v):])) / sum(weights[-len(v):])
        
        if weighted > 0.6:
            return "Tài", 68, "W-Recent"
        elif weighted < 0.4:
            return "Xỉu", 68, "W-Recent"
        return None, 0, f"W{weighted:.2f}"

class ClusteringModel:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0, "Need data"
        s = encode(tx_list)
        recent8 = s[-8:]
        
        best_match = None
        best_score = 0
        for i in range(len(s)-16):
            segment = s[i:i+8]
            score = sum(a==b for a,b in zip(recent8, segment)) / 8
            if score > best_score and score > 0.75:
                best_score = score
                if i+8 < len(s):
                    best_match = s[i+8]
        
        if best_match:
            return decode(best_match), min(best_score*100, 80), f"Cluster{best_score:.0%}"
        return None, 0, "Cluster TB"

# ================= ENSEMBLE =================

class SuperEnsemble:
    def __init__(self):
        self.models = {
            "Markov": MarkovChain(),
            "N-Gram": NGramModel(),
            "Pattern": PatternModel(),
            "Streak": StreakModel(),
            "Reversal": ReversalModel(),
            "Cycle": CycleModel(),
            "Momentum": MomentumModel(),
            "Trend": TrendModel(),
            "MA": MAModel(),
            "Bayes": BayesianModel(),
            "Entropy": EntropyModel(),
            "MarkovHO": MarkovHighOrder(),
            "Regression": RegressionModel(),
            "FreqAdapt": FreqAdaptiveModel(),
            "DeepPattern": DeepPatternModel(),
            "Fibonacci": FibonacciModel(),
            "GoldenRatio": GoldenRatioModel(),
            "Volatility": VolatilityModel(),
            "SupportResist": SupportResistanceModel(),
            "TimeSeries": TimeSeriesModel(),
            "Clustering": ClusteringModel()
        }
        self.weights = {name: 1.0 for name in self.models}
        self.performance = {name: {"dung": 0, "sai": 0} for name in self.models}
        self.load_weights()
    
    def load_weights(self):
        try:
            if os.path.exists(WEIGHTS_FILE):
                with open(WEIGHTS_FILE, "r") as f:
                    data = json.load(f)
                    for k, v in data.get("weights", {}).items():
                        if k in self.weights:
                            self.weights[k] = v
        except:
            pass
    
    def save_weights(self):
        try:
            with open(WEIGHTS_FILE, "w") as f:
                json.dump({"weights": self.weights, "performance": self.performance}, f)
        except:
            pass
    
    def update(self, actual, tx_list_before):
        if len(tx_list_before) < 5:
            return
        for name, model in self.models.items():
            try:
                result = model.predict(list(tx_list_before))
                if result and len(result) >= 2 and result[0]:
                    pred = result[0]
                    if pred == actual:
                        self.performance[name]["dung"] += 1
                        self.weights[name] = min(self.weights[name] * 1.06, 3.0)
                    else:
                        self.performance[name]["sai"] += 1
                        self.weights[name] = max(self.weights[name] * 0.94, 0.3)
            except:
                pass
        self.save_weights()
        save_data()
    
    def predict(self, tx_list):
        votes = Counter()
        details = {}
        reasons = {}
        
        for name, model in self.models.items():
            try:
                result = model.predict(tx_list)
                if result and len(result) >= 2 and result[0]:
                    pred, conf = result[0], result[1]
                    if conf < 55:  # Chỉ nhận tín hiệu mạnh
                        continue
                    weight = self.weights.get(name, 1.0)
                    votes[pred] += conf * weight
                    details[name] = round(conf, 1)
                    reasons[name] = result[2] if len(result) > 2 else ""
            except:
                pass
        
        if not votes:
            # Fallback thông minh: đảo nếu bệt, tiếp nếu không
            if tx_list:
                s = encode(tx_list)
                streak = 0
                last = s[-1]
                for c in reversed(s):
                    if c == last: streak += 1
                    else: break
                if streak >= 2:
                    fallback = "Xỉu" if tx_list[-1] == "Tài" else "Tài"
                    return fallback, 52, {}, {}, "Fallback đảo"
                return decode(s[-1]), 52, {}, {}, "Fallback tiếp"
            return "Tài", 50, {}, {}, "Random"
        
        winner = votes.most_common(1)[0]
        pred = winner[0]
        total = sum(votes.values())
        conf = min(winner[1]/total*100, 90) if total else 50
        
        top = sorted([(k, v) for k, v in reasons.items() if v],
                    key=lambda x: self.weights.get(x[0], 1), reverse=True)[:5]
        reason_str = " | ".join([f"{k}:{v}" for k, v in top])
        
        return pred, round(conf, 1), details, self.weights.copy(), reason_str

ai_engine = SuperEnsemble()
load_data()

# ================= PREDICT =================
def predict(tx_list):
    if len(tx_list) < MIN_PHIEN:
        return "Chờ dữ liệu", "0%", {}, {}, "Chưa đủ 15 phiên"
    pred, conf, details, weights, reason = ai_engine.predict(tx_list)
    return pred, f"{conf}%", details, weights, reason

# ================= UPDATE STATS =================
def update_stats(actual, phien_id):
    global _prev_pred, stats, lich_su
    if not _prev_pred or _prev_pred == "Chờ dữ liệu":
        return
    dung = (_prev_pred == actual)
    stats["tong"] += 1
    if dung:
        stats["dung"] += 1
        stats["cd"] += 1
        stats["cs"] = 0
        stats["max_cd"] = max(stats["max_cd"], stats["cd"])
    else:
        stats["sai"] += 1
        stats["cs"] += 1
        stats["cd"] = 0
        stats["max_cs"] = max(stats["max_cs"], stats["cs"])
    
    lich_su.append({
        "phien": phien_id,
        "du_doan": _prev_pred,
        "ket_qua": actual,
        "dung": "✅" if dung else "❌",
        "time": datetime.now().strftime("%H:%M:%S")
    })
    if len(lich_su) > 200:
        lich_su.pop(0)

# ================= TÍNH CHUỖI =================
def tinh_chuoi(tx_list):
    if not tx_list:
        return 0, None, 0, 0
    s = encode(tx_list)
    max_tai = max_xiu = cur_tai = cur_xiu = 0
    current_streak = 0
    current_type = None
    for i, c in enumerate(s):
        if c == "T":
            cur_tai += 1
            cur_xiu = 0
            max_tai = max(max_tai, cur_tai)
        else:
            cur_xiu += 1
            cur_tai = 0
            max_xiu = max(max_xiu, cur_xiu)
        if i == len(s) - 1:
            current_streak = cur_tai if c == "T" else cur_xiu
            current_type = decode(c)
    return current_streak, current_type, max_tai, max_xiu

# ================= GET DATA =================
def get_data():
    for url in API_LIST:
        try:
            res = requests.get(url, timeout=3)
            if res.status_code != 200:
                continue
            data = res.json()
            if "data" not in data:
                continue
            info = data["data"]
            phien = int(info.get("Expect", 0))
            dice = [int(x) for x in info.get("OpenCode", "0,0,0").split(",")]
            tong = sum(dice)
            return phien, dice, tong
        except:
            continue
    return None

# ================= BACKGROUND =================
def background():
    global last_result, _prev_pred, history_tx, history_pt, history_id
    
    last_phien = None
    
    while True:
        data = get_data()
        
        if data:
            phien, dice, tong = data
            
            if phien != last_phien:
                ket = "Tài" if tong >= 11 else "Xỉu"
                tx = "T" if ket == "Tài" else "X"
                
                # Huấn luyện trước
                if len(history_tx) >= MIN_PHIEN and _prev_pred and _prev_pred != "Chờ dữ liệu":
                    ai_engine.update(ket, list(history_tx))
                
                update_stats(tx, phien)
                
                history_tx.append(ket)
                history_pt.append(tong)
                history_id.append(phien)
                
                # Tính chuỗi
                streak, stype, max_tai, max_xiu = tinh_chuoi(list(history_tx))
                
                # AI Predict
                tx_list = list(history_tx)
                du_doan, do_tin_cay, model_confs, weights, phan_tich = predict(tx_list)
                _prev_pred = du_doan
                
                # Pattern
                pattern = encode(tx_list)[-25:] if len(tx_list) >= 25 else encode(tx_list)
                
                # Tỷ lệ thật
                td = stats["tong"]
                ty_le = f"{stats['dung']/td*100:.1f}%" if td else "0%"
                
                last_result = {
                    "status": "success",
                    "data": {
                        "Phien": phien,
                        "Xuc_xac_1": dice[0],
                        "Xuc_xac_2": dice[1],
                        "Xuc_xac_3": dice[2],
                        "Tong": tong,
                        "Ket": ket,
                        "Phien_hien_tai": phien + 1,
                        "Du_doan": du_doan,
                        "Do_tin_cay": do_tin_cay,
                        "Pattern": pattern,
                        "Max_chuoi_Tai": max_tai,
                        "Max_chuoi_Xiu": max_xiu,
                        "Ty_le_dung": ty_le,
                        "AI_Models": model_confs,
                        "Trong_so": {k: round(v, 2) for k, v in weights.items()},
                        "Phan_tich": phan_tich,
                      
                    }
                }
                
               
                print("\n" + "=" * 70)
                print(f"🎲 PHIÊN {phien} | {dice} = {tong} [{ket}]")
                print("-" * 70)
                print(f"Pattern  : {pattern}")
                print(f"Chuỗi    : {stype} x{streak}" if stype else "Chuỗi    : N/A")
                print(f"🔮 Dự đoán: >>> {du_doan} <<< ({do_tin_cay})")
                print(f"📊 Tỷ lệ  : {ty_le} ({stats['dung']}/{td})")
                print(f"🧠 Phân tích: {phan_tich}")
                if model_confs:
                    active = {k: v for k, v in model_confs.items() if v > 0}
                    print(f"🤖 AI ({len(active)}/20 models): {active}")
                print("=" * 70)
                
                last_phien = phien
        
        time.sleep(2)

# ================= API =================
@app.route("/api/taixiumd5")
def api_main():
    return jsonify(last_result if last_result else {"status": "waiting"})

@app.route("/api/lichsu")
def api_history():
    td = stats["tong"]
    return jsonify({
        "tong": td,
        "dung": stats["dung"],
        "sai": stats["sai"],
        "ty_le": f"{stats['dung']/td*100:.1f}%" if td else "0%",
        "max_cd": stats["max_cd"],
        "max_cs": stats["max_cs"],
        "chuoi_hien_tai": f"{stats['cd']} dung" if stats["cd"] > 0 else (f"{stats['cs']} sai" if stats["cs"] > 0 else "0"),
        "lich_su": lich_su[-30:],
        "lich_su_day_du": lich_su[-100:],
        "ai_weights": {k: round(v, 2) for k, v in ai_engine.weights.items()},
        "ai_performance": ai_engine.performance
    })

@app.route("/")
def home():
    return jsonify({"status": "ok", "data": last_result.get("data", {})})

@app.route("/api/thongke")
def api_thongke():
    tx_list = list(history_tx)
    if len(tx_list) < 10:
        return jsonify({"error": "Chua du du lieu"})
    s = encode(tx_list)
    return jsonify({
        "tong_phien": len(tx_list),
        "tai": s.count("T"),
        "xiu": s.count("X"),
        "tai_rate": f"{s.count('T')/len(s)*100:.1f}%",
        "pattern_20": s[-20:],
        "pattern_50": s[-50:] if len(s) >= 50 else s,
        "max_tai": stats["max_cd"],
        "max_xiu": stats["max_cs"],
        "ai_active": len([m for m in ai_engine.weights.values() if m > 0.5])
    })

# ================= RUN =================
if __name__ == "__main__":
    threading.Thread(target=background, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
