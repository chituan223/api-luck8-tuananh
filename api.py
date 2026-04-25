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
MIN_PHIEN = 20
MAX_HISTORY = 200
DATA_FILE = "ai_training_data.json"
WEIGHTS_FILE = "ai_weights.json"

API_LIST = [
    "https://luck8bot.com/api/GetNewLottery/TaixiuMd5?id=",
    "http://luck8bot.com/api/GetNewLottery/TaixiuMd5?id="
]

# ================= DATA =================
history_tx = deque(maxlen=MAX_HISTORY)
history_pt = deque(maxlen=MAX_HISTORY)
history_id = deque(maxlen=MAX_HISTORY)
history_dice = deque(maxlen=MAX_HISTORY)

lich_su = []
stats = {
    "tong": 0, "dung": 0, "sai": 0,
    "cd": 0, "cs": 0, "max_cd": 0, "max_cs": 0
}

last_result = {}
_prev_pred = None
_training_data = []
_model_performance = {}

# ================= UTILS =================
def encode(tx_list):
    return "".join(["T" if x == "Tài" else "X" for x in tx_list])

def decode(c):
    return "Tài" if c == "T" else "Xỉu"

def save_training():
    """Lưu dữ liệu huấn luyện"""
    try:
        data = {
            "history": list(history_tx),
            "points": list(history_pt),
            "ids": list(history_id),
            "dice": [list(d) for d in history_dice],
            "performance": _model_performance
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

def load_training():
    """Tải dữ liệu huấn luyện"""
    global _training_data, _model_performance
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                for h, p, i, d in zip(data.get("history", []), 
                                      data.get("points", []),
                                      data.get("ids", []),
                                      data.get("dice", [])):
                    history_tx.append(h)
                    history_pt.append(p)
                    history_id.append(i)
                    history_dice.append(tuple(d))
                _model_performance = data.get("performance", {})
    except:
        pass

# ================= 15 AI MODELS =================

# 1. MARKOV CHAIN
class MarkovChain:
    def predict(self, tx_list, order=3):
        if len(tx_list) < order + 5:
            return None, 0
        s = encode(tx_list)
        trans = {}
        for i in range(len(s) - order):
            state = s[i:i+order]
            nxt = s[i+order]
            trans.setdefault(state, Counter())[nxt] += 1
        
        current = s[-order:]
        if current not in trans:
            return None, 0
        
        counts = trans[current]
        total = sum(counts.values())
        p_t = counts.get("T", 0) / total
        p_x = counts.get("X", 0) / total
        pred = "Tài" if p_t > p_x else "Xỉu"
        return pred, max(p_t, p_x) * 100

# 2. N-GRAM MODEL
class NGramModel:
    def predict(self, tx_list, max_n=6):
        if len(tx_list) < 10:
            return None, 0
        s = encode(tx_list)
        votes = Counter()
        confs = []
        
        for n in range(2, min(max_n+1, len(s)//2+1)):
            grams = Counter()
            for i in range(len(s)-n):
                grams[(s[i:i+n], s[i+n])] += 1
            
            cur = s[-n:]
            t_score = sum(v for (g,c),v in grams.items() if g==cur and c=="T")
            x_score = sum(v for (g,c),v in grams.items() if g==cur and c=="X")
            total = t_score + x_score
            
            if total > 0:
                if t_score > x_score:
                    votes["Tài"] += n
                    confs.append(t_score/total*100)
                elif x_score > t_score:
                    votes["Xỉu"] += n
                    confs.append(x_score/total*100)
        
        if not votes:
            return None, 0
        pred = votes.most_common(1)[0][0]
        return pred, min(sum(confs)/len(confs), 95) if confs else 50

# 3. PATTERN DETECTION
class PatternModel:
    def predict(self, tx_list):
        if len(tx_list) < 5:
            return None, 0
        s = encode(tx_list)
        n = len(s)
        
        # Bệt
        streak = 0
        last = s[-1]
        for c in reversed(s):
            if c == last: streak += 1
            else: break
        if streak >= 3:
            opp = "X" if last=="T" else "T"
            return decode(opp), min(50+streak*10, 95), f"Bệt{streak}"
        
        # 1-1
        if n >= 4:
            recent = s[-6:]
            if all(recent[i]!=recent[i+1] for i in range(len(recent)-1)):
                return decode("X" if recent[-1]=="T" else "T"), 80, "1-1"
        
        # 2-2
        if n >= 6:
            recent = s[-8:]
            is22 = True
            for i in range(0, min(6, len(recent)-3), 2):
                if not (recent[i]==recent[i+1] and recent[i+2]==recent[i+3] and recent[i]!=recent[i+2]):
                    is22 = False
                    break
            if is22:
                return decode(recent[0] if len(recent)%4==0 else recent[2]), 85, "2-2"
        
        # 3-3
        if n >= 9:
            recent = s[-12:]
            is33 = True
            for i in range(0, min(9, len(recent)-5), 3):
                if not (recent[i]==recent[i+1]==recent[i+2] and recent[i+3]==recent[i+4]==recent[i+5] and recent[i]!=recent[i+3]):
                    is33 = False
                    break
            if is33:
                return decode(recent[0]), 90, "3-3"
        
        # Cân bằng
        if n >= 20:
            recent = s[-20:]
            t_pct = recent.count("T")/20
            if t_pct > 0.7:
                return "Xỉu", min(50+(t_pct-0.5)*100, 90), f"Tài{t_pct:.0%}"
            elif t_pct < 0.3:
                return "Tài", min(50+(0.5-t_pct)*100, 90), f"Xỉu{1-t_pct:.0%}"
        
        return None, 0, "Không pattern"

# 4. STREAK ANALYSIS
class StreakModel:
    def predict(self, tx_list):
        if len(tx_list) < 10:
            return None, 0
        s = encode(tx_list)
        streaks = []
        cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]: cur += 1
            else:
                streaks.append(cur)
                cur = 1
        streaks.append(cur)
        
        avg = sum(streaks)/len(streaks)
        current = streaks[-1]
        
        if current > avg + 0.5:
            opp = "Xỉu" if s[-1]=="T" else "Tài"
            return opp, min(50+(current-avg)*20, 90), f"Streak{current}>avg{avg:.1f}"
        elif current <= avg:
            return decode(s[-1]), min(50+(avg-current)*15, 85), f"Streak{current}<=avg{avg:.1f}"
        return None, 0, "Streak TB"

# 5. REVERSAL DETECTION
class ReversalModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        s = encode(tx_list)[-15:]
        rev = sum(1 for i in range(1, len(s)) if s[i]!=s[i-1])
        rate = rev/(len(s)-1)
        
        if rate > 0.6:
            return decode("X" if s[-1]=="T" else "T"), min(50+rate*30, 85), f"Đảo{rate:.0%}"
        elif rate < 0.3:
            return decode(s[-1]), min(50+(0.3-rate)*100, 80), f"ÍtĐảo{rate:.0%}"
        return None, 0, f"ĐảoTB{rate:.0%}"

# 6. CYCLE PREDICTION
class CycleModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        s = encode(tx_list)
        best_cycle = None
        best_score = 0
        
        for c_len in range(2, min(12, len(s)//2)):
            matches = sum(1 for i in range(c_len, len(s)) if s[i]==s[i-c_len])
            score = matches/(len(s)-c_len)
            if score > best_score and score > 0.55:
                best_score = score
                best_cycle = c_len
        
        if best_cycle:
            return decode(s[-best_cycle]), min(best_score*100, 90), f"ChuKỳ{best_cycle}"
        return None, 0, "Không chu kỳ"

# 7. MOMENTUM MODEL
class MomentumModel:
    def predict(self, tx_list):
        if len(tx_list) < 10:
            return None, 0
        s = encode(tx_list)
        momentums = []
        for w in [5, 10, 15]:
            if len(s) >= w:
                momentums.append(s[-w:].count("T")/w)
        
        if not momentums:
            return None, 0
        avg = sum(momentums)/len(momentums)
        
        if avg > 0.6:
            return "Tài", min(avg*100, 90), f"MomTài{avg:.0%}"
        elif avg < 0.4:
            return "Xỉu", min((1-avg)*100, 90), f"MomXỉu{1-avg:.0%}"
        return None, 0, f"MomTB{avg:.0%}"

# 8. TREND STRENGTH
class TrendModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        s = encode(tx_list)
        half = len(s)//2
        f = s[:half].count("T")/half if half else 0.5
        se = s[half:].count("T")/(len(s)-half) if len(s)-half else 0.5
        diff = abs(se-f)
        
        if diff > 0.15:
            return "Tài" if se>f else "Xỉu", min(50+diff*200, 85), f"Trend{'Tài' if se>f else 'Xỉu'}"
        return None, 0, f"TrendYếu{diff:.0%}"

# 9. MOVING AVERAGE
class MAModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        v = [1 if x=="Tài" else 0 for x in tx_list]
        ma5 = sum(v[-5:])/5
        ma10 = sum(v[-10:])/10
        ma20 = sum(v[-20:])/20 if len(v)>=20 else ma10
        
        if ma5 > ma10 > ma20:
            return "Tài", 78, "MA Bull"
        elif ma5 < ma10 < ma20:
            return "Xỉu", 78, "MA Bear"
        
        if len(v) >= 6:
            pma5 = sum(v[-6:-1])/5
            if ma5 > pma5 and ma5 > 0.5:
                return "Tài", 72, "MA5↑"
            elif ma5 < pma5 and ma5 < 0.5:
                return "Xỉu", 72, "MA5↓"
        return None, 0, f"MA5={ma5:.2f}"

# 10. BAYESIAN MODEL
class BayesianModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        s = encode(tx_list)
        
        # Prior
        prior_t = s.count("T")/len(s)
        prior_x = 1 - prior_t
        
        # Likelihood dựa trên 5 phiên gần nhất
        recent5 = s[-5:]
        if recent5.count("T") >= 4:
            likelihood_t = 0.3
            likelihood_x = 0.7
        elif recent5.count("X") >= 4:
            likelihood_t = 0.7
            likelihood_x = 0.3
        else:
            return None, 0, "Bayes TB"
        
        # Posterior
        post_t = likelihood_t * prior_t
        post_x = likelihood_x * prior_x
        total = post_t + post_x
        
        if total == 0:
            return None, 0
        
        p_t = post_t/total
        pred = "Tài" if p_t > 0.5 else "Xỉu"
        conf = max(p_t, 1-p_t)*100
        return pred, min(conf, 90), f"Bayes{'Tài' if p_t>0.5 else 'Xỉu'}"

# 11. ENTROPY MODEL
class EntropyModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        s = encode(tx_list)[-20:]
        t_count = s.count("T")
        x_count = len(s) - t_count
        
        if t_count == 0 or x_count == 0:
            return None, 0
        
        p_t = t_count/len(s)
        p_x = x_count/len(s)
        entropy = -(p_t*math.log2(p_t) + p_x*math.log2(p_x))
        
        # Entropy thấp = có pattern, entropy cao = random
        if entropy < 0.7:
            # Có pattern mạnh -> tiếp tục trend
            return decode(s[-1]), 75, f"EntropyThấp{entropy:.2f}"
        elif entropy > 0.95:
            # Random -> đảo
            return decode("X" if s[-1]=="T" else "T"), 60, f"EntropyCao{entropy:.2f}"
        return None, 0, f"EntropyTB{entropy:.2f}"

# 12. MARKOV HIGHER ORDER
class MarkovHighOrder:
    def predict(self, tx_list):
        if len(tx_list) < 30:
            return None, 0
        
        # Thử nhiều order, chọn tốt nhất
        best_pred = None
        best_conf = 0
        
        for order in [2, 3, 4, 5]:
            if len(tx_list) < order + 10:
                continue
            s = encode(tx_list)
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
            
            p_t = counts.get("T", 0)/total
            conf = max(p_t, 1-p_t)*100
            if conf > best_conf:
                best_conf = conf
                best_pred = "Tài" if p_t > 0.5 else "Xỉu"
        
        if best_pred:
            return best_pred, min(best_conf, 95), f"MarkovHO"
        return None, 0, "MarkovHO TB"

# 13. REGRESSION SIMPLE
class RegressionModel:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0
        
        # Chuyển thành số và tính xu hướng
        v = [1 if x=="Tài" else 0 for x in tx_list[-20:]]
        n = len(v)
        
        # Linear regression đơn giản
        sum_x = sum(range(n))
        sum_y = sum(v)
        sum_xy = sum(i*v[i] for i in range(n))
        sum_x2 = sum(i*i for i in range(n))
        
        denom = n*sum_x2 - sum_x*sum_x
        if denom == 0:
            return None, 0
        
        slope = (n*sum_xy - sum_x*sum_y) / denom
        
        if slope > 0.05:
            return "Tài", min(50+slope*500, 85), f"Reg↑{slope:.3f}"
        elif slope < -0.05:
            return "Xỉu", min(50+abs(slope)*500, 85), f"Reg↓{slope:.3f}"
        return None, 0, f"RegTB{slope:.3f}"

# 14. FREQUENCY ADAPTIVE
class FreqAdaptiveModel:
    def predict(self, tx_list):
        if len(tx_list) < 30:
            return None, 0
        
        # Phân tích tần suất theo thời gian
        s = encode(tx_list)
        windows = [10, 20, 30]
        signals = []
        
        for w in windows:
            if len(s) < w:
                continue
            recent = s[-w:]
            t_pct = recent.count("T")/w
            
            if t_pct > 0.65:
                signals.append(("Xỉu", t_pct))
            elif t_pct < 0.35:
                signals.append(("Tài", 1-t_pct))
        
        if not signals:
            return None, 0, "Freq TB"
        
        # Lấy signal mạnh nhất
        signals.sort(key=lambda x: x[1], reverse=True)
        pred, conf = signals[0]
        return pred, min(conf*100, 90), f"Freq{conf:.0%}"

# 15. DEEP PATTERN (Soi sâu nhất)
class DeepPatternModel:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0
        
        s = encode(tx_list)
        
        # Tìm pattern phức tạp: lặp lại sau khoảng cách
        best_pattern = None
        best_score = 0
        
        for pat_len in range(3, min(8, len(s)//3)):
            for gap in range(1, min(6, (len(s)-pat_len)//2)):
                matches = 0
                total = 0
                for i in range(len(s)-pat_len-gap):
                    if s[i:i+pat_len] == s[i+gap:i+gap+pat_len]:
                        # Kiểm tra kết quả sau pattern
                        if i+pat_len+gap < len(s) and i+pat_len < len(s):
                            matches += 1
                        total += 1
                
                score = matches/total if total else 0
                if score > best_score and score > 0.6:
                    best_score = score
                    best_pattern = (pat_len, gap)
        
        if best_pattern:
            pat_len, gap = best_pattern
            # Dự đoán dựa trên pattern tìm được
            recent = s[-pat_len:]
            for i in range(len(s)-pat_len-gap, -1, -1):
                if s[i:i+pat_len] == recent:
                    if i+pat_len+gap < len(s):
                        return decode(s[i+pat_len+gap]), min(best_score*100, 90), f"DeepPat{pat_len}/{gap}"
                    break
        
        # Fallback: phân tích chuỗi con lặp
        for length in range(4, min(10, len(s)//2)):
            sub = s[-length:]
            count = s.count(sub)
            if count >= 3:
                # Tìm vị trí xuất hiện và kết quả sau
                positions = [i for i in range(len(s)-length) if s[i:i+length]==sub]
                if len(positions) >= 2:
                    next_chars = [s[p+length] for p in positions if p+length < len(s)]
                    if next_chars:
                        t_count = next_chars.count("T")
                        pred = "Tài" if t_count > len(next_chars)/2 else "Xỉu"
                        conf = max(t_count, len(next_chars)-t_count)/len(next_chars)*100
                        return pred, min(conf, 90), f"DeepSub{length}"
        
        return None, 0, "Deep TB"

# ================= ENSEMBLE & ADAPTIVE =================

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
            "DeepPattern": DeepPatternModel()
        }
        self.weights = {name: 1.0 for name in self.models}
        self.history = {name: {"dung": 0, "sai": 0} for name in self.models}
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
                json.dump({"weights": self.weights, "history": self.history}, f)
        except:
            pass
    
    def update(self, actual, tx_list_before):
        """Huấn luyện: cập nhật weights dựa trên kết quả thực tế"""
        if len(tx_list_before) < 5:
            return
        
        for name, model in self.models.items():
            try:
                result = model.predict(list(tx_list_before))
                if result and len(result) >= 2:
                    pred = result[0]
                    if pred == actual:
                        self.history[name]["dung"] += 1
                        self.weights[name] = min(self.weights[name] * 1.08, 5.0)
                    else:
                        self.history[name]["sai"] += 1
                        self.weights[name] = max(self.weights[name] * 0.92, 0.2)
            except:
                pass
        
        self.save_weights()
        save_training()
    
    def predict(self, tx_list):
        votes = Counter()
        details = {}
        reasons = {}
        
        for name, model in self.models.items():
            try:
                result = model.predict(tx_list)
                if result and len(result) >= 2:
                    pred, conf = result[0], result[1]
                    weight = self.weights.get(name, 1.0)
                    votes[pred] += conf * weight
                    details[name] = round(conf, 1)
                    reasons[name] = result[2] if len(result) > 2 else ""
            except Exception as e:
                continue
        
        if not votes:
            fallback = "Xỉu" if tx_list and tx_list[-1]=="Tài" else "Tài"
            return fallback, 50, {}, {}, "Fallback đảo"
        
        winner = votes.most_common(1)[0]
        pred = winner[0]
        total = sum(votes.values())
        conf = min(winner[1]/total*100, 95) if total else 50
        
        # Top 5 reasons
        top_reasons = sorted([(k, v) for k, v in reasons.items() if v], 
                            key=lambda x: self.weights.get(x[0], 1), reverse=True)[:5]
        reason_str = " | ".join([f"{k}:{v}" for k, v in top_reasons])
        
        return pred, round(conf, 1), details, self.weights.copy(), reason_str

# Khởi tạo AI
ai_engine = SuperEnsemble()
load_training()

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
        "thoi_gian": datetime.now().strftime("%H:%M:%S")
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
    global last_result, _prev_pred, history_tx, history_pt, history_id, history_dice
    
    last_phien = None
    
    while True:
        data = get_data()
        
        if data:
            phien, dice, tong = data
            
            if phien != last_phien:
                ket = "Tài" if tong >= 11 else "Xỉu"
                tx = "T" if ket == "Tài" else "X"
                
                # Huấn luyện trước khi cập nhật
                if len(history_tx) >= MIN_PHIEN and _prev_pred and _prev_pred != "Chờ dữ liệu":
                    ai_engine.update(ket, list(history_tx))
                
                update_stats(ket, phien)
                
                # Lưu history
                history_tx.append(ket)
                history_pt.append(tong)
                history_id.append(phien)
                history_dice.append(tuple(dice))
                
                # Tính chuỗi
                streak, stype, max_tai, max_xiu = tinh_chuoi(list(history_tx))
                
                # AI PREDICT
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
                        "So_phien_hoc": len(tx_list),
                        
                    }
                }
                
                # Log
                print("\n" + "=" * 70)
                print(f"🎲 PHIÊN {phien} | {dice[0]}-{dice[1]}-{dice[2]} = {tong} [{ket}]")
                print("-" * 70)
                print(f"Pattern  : {pattern}")
                print(f"Chuỗi    : {stype} x{streak}" if stype else "Chuỗi    : N/A")
                print(f"🔮 Dự đoán: >>> {du_doan} <<< ({do_tin_cay})")
                print(f"📊 Tỷ lệ  : {ty_le} ({stats['dung']}/{td})")
                print(f"🧠 Phân tích: {phan_tich}")
                if model_confs:
                    active = {k: v for k, v in model_confs.items() if v > 0}
                    print(f"🤖 AI ({len(active)} models): {active}")
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
        "chuoi_hien_tai": f"{stats['cd']} đúng" if stats["cd"] > 0 else (f"{stats['cs']} sai" if stats["cs"] > 0 else "0"),
        "lich_su": lich_su[-30:],
        "lich_su_day_du": lich_su[-100:],
        "ai_weights": {k: round(v, 2) for k, v in ai_engine.weights.items()},
        "ai_performance": ai_engine.history
    })

@app.route("/")
def home():
    return jsonify({"status": "ok", "data": last_result.get("data", {})})

@app.route("/api/thongke")
def api_thongke():
    """API thống kê chi tiết"""
    tx_list = list(history_tx)
    if len(tx_list) < 10:
        return jsonify({"error": "Chưa đủ dữ liệu"})
    
    s = encode(tx_list)
    return jsonify({
        "tong_so_phien": len(tx_list),
        "tai_count": s.count("T"),
        "xiu_count": s.count("X"),
        "tai_rate": f"{s.count('T')/len(s)*100:.1f}%",
        "pattern_20": s[-20:],
        "pattern_50": s[-50:] if len(s) >= 50 else s,
        "max_chuoi_tai": stats["max_cd"],
        "max_chuoi_xiu": stats["max_cs"],
        "ai_models_active": len([m for m in ai_engine.weights.values() if m > 0.5])
    })

# ================= RUN =================
if __name__ == "__main__":
    threading.Thread(target=background, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
