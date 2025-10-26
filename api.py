from flask import Flask, jsonify
import requests

app = Flask(__name__)

PORT = 3000

# Lưu pattern gần nhất (tối đa 50 phiên)
pattern_history = []

MAX_HISTORY = 50

def update_pattern(result):
    global pattern_history
    pattern_history.append(result)
    if len(pattern_history) > MAX_HISTORY:
        pattern_history.pop(0)

# ================= 10 thuật toán cơ bản =================

def algo1_weightedRecent(hist):
    if not hist: return "Tài"
    t_score = x_score = 0
    n = len(hist)
    for i, v in enumerate(hist):
        weight = (i+1)/n
        if v=="Tài": t_score+=weight
        else: x_score+=weight
    return "Tài" if t_score>=x_score else "Xỉu"

def algo2_expDecay(hist, decay=0.6):
    if not hist: return "Tài"
    t = x = 0
    w = 1
    for v in reversed(hist):
        if v=="Tài": t+=w
        else: x+=w
        w*=decay
    return "Tài" if t>x else "Xỉu"

def algo3_longChainReverse(hist, k=3):
    if not hist: return "Tài"
    last = hist[-1]
    chain = 1
    for v in reversed(hist[:-1]):
        if v==last: chain+=1
        else: break
    if chain>=k: return "Xỉu" if last=="Tài" else "Tài"
    return last

def algo4_windowMajority(hist, window=5):
    if not hist: return "Tài"
    win = hist[-window:] if len(hist)>=window else hist
    return "Tài" if win.count("Tài")>=len(win)/2 else "Xỉu"

def algo5_alternation(hist):
    if len(hist)<4: return "Tài"
    flips = sum(1 for i in range(1,4) if hist[-i]!=hist[-i-1])
    if flips>=3: return "Xỉu" if hist[-1]=="Tài" else "Tài"
    return hist[-1]

def algo6_patternRepeat(hist):
    L = len(hist)
    if L<4: return "Tài"
    for length in range(2, min(6, L//2)+1):
        a = "".join(hist[-length:])
        b = "".join(hist[-2*length:-length])
        if a==b: return hist[-length]
    return algo4_windowMajority(hist,4)

def algo7_mirror(hist):
    if len(hist)<8: return hist[-1] if hist else "Tài"
    if hist[-4:]==hist[-8:-4]:
        return "Xỉu" if hist[-1]=="Tài" else "Tài"
    return hist[-1]

def algo8_entropy(hist):
    if not hist: return "Tài"
    t = hist.count("Tài")
    x = len(hist)-t
    diff = abs(t-x)
    if diff<=len(hist)//5: return "Xỉu" if hist[-1]=="Tài" else "Tài"
    return "Xỉu" if t>x else "Tài"

def algo9_momentum(hist):
    if len(hist)<2: return "Tài"
    score = sum(1 if hist[i]==hist[i-1] else -1 for i in range(1,len(hist)))
    return hist[-1] if score>0 else ("Xỉu" if hist[-1]=="Tài" else "Tài")

def algo10_freqRatio(hist):
    if not hist: return "Tài"
    ratio = hist.count("Tài")/len(hist)
    if ratio>0.62: return "Xỉu"
    if ratio<0.38: return "Tài"
    return hist[-1]

algos = [
    algo1_weightedRecent, algo2_expDecay, algo3_longChainReverse,
    algo4_windowMajority, algo5_alternation, algo6_patternRepeat,
    algo7_mirror, algo8_entropy, algo9_momentum, algo10_freqRatio
]

# ================= API =================

def get_taixiu(total):
    return "Tài" if total>=11 else "Xỉu"

@app.route("/api/taixiu/", methods=["GET"])
def taixiu_lottery():
    global pattern_history
    try:
        resp = requests.get("https://1.bot/GetNewLottery/LT_TaixiuMD5", timeout=10)
        json_data = resp.json()
        if not json_data or json_data.get("state") !=1:
            return jsonify({"error":"Dữ liệu không hợp lệ"}),500

        data = json_data["data"]
        dice = list(map(int, data["OpenCode"].split(",")))
        d1,d2,d3 = dice
        total = sum(dice)
        ket_qua = get_taixiu(total)
        update_pattern(ket_qua)

        # Dự đoán bằng 10 thuật toán
        results = [algo(pattern_history) for algo in algos]
        tai_count = results.count("Tài")
        xiu_count = results.count("Xỉu")
        du_doan = "Tài" if tai_count>=xiu_count else "Xỉu"
        do_tin_cay = round(max(tai_count,xiu_count)/len(algos)*100,2)

        return jsonify({
            "id":"tuananh",
            "Phien":data["Expect"],
            "Xuc_xac_1":d1,
            "Xuc_xac_2":d2,
            "Xuc_xac_3":d3,
            "Tong":total,
            "Ket_qua":ket_qua,
            "Pattern":pattern_history,
            "Du_doan":du_doan,
            "Do_tin_cay":f"{do_tin_cay}%",
            "Chi_tiet_algos": results
        })

    except Exception as e:
        return jsonify({"error":"Lỗi khi fetch dữ liệu","details":str(e)}),500

if __name__=="__main__":
    print(f"✅ Server đang chạy tại http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT)