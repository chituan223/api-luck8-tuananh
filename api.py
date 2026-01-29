from flask import Flask, jsonify
import threading
import websocket
import json
import time
from collections import deque

# ================== CONFIG ==================
WS_URL = "wss://taixiumd5.system32-cloudfare-356783752985678522.monster/signalr/reconnect?transport=webSockets&connectionToken=SgIYXqnbkJRw6FvkcaXYVrAcj9Rkcx758qlxIanF3odMFBbrqY%2BJJ%2FVvZUnOX0Z2pNFJwckC2pCxXefKhAclClEefIExyEGKc9Z6zfoZsoa9oUAzcs1LNw2G3jxr7w9j&connectionData=%5B%7B%22name%22%3A%22md5luckydiceHub%22%7D%5D&tid=6&access_token=05%2F7JlwSPGzg4ARi0d7%2FLOcNQQ%2BecAvgB3UwDAmuWFJiZj%2Blw1TcJ0PZt5VeUAHKLVCmODRrV5CHPNbit3mc868w8zYBuyQ5Xlu1AZVsEElr9od2qJ8S9N2GLAdQnd0VL8fj8IAGPMsP45pdIIXZysKmRi40b%2FOVLAp4yOpkaXP3icyn2%2Fodm397vVKSY9AlMCcH15AghVm3lx5JM%2BoUuP%2Fkjgh5xWXtdTQkd9W3%2BQBY25AdX3CvOZ2I17r67METGpFv8cP7xmAoySWEnokU2IcOKu3mzvRWXsG7N5sHFkv%2FIKw%2F1IPCNY2oi8RygWpHwIFWcHGdeoTeM6kskfrqNSmhapPBCREit0So1HOC6jOiz5IyKVNadwp8EfsxKzBOKE0z0zdavvY6wXrSZhIJeIqKqVAt3SEuoG82a%2BjwxNo%3D.5a1d88795043d5c4ef6538c9edfb5ff93e65b852d89b71344bdd5ec80eb63e24"
PING_INTERVAL = 15
MAX_HISTORY = 100
FILE_NAME = "thuat_toan_tai_xiu.txt"

app = Flask(__name__)
lock = threading.Lock()

# ================== DATA STORE ==================
results_history = deque(maxlen=MAX_HISTORY)

latest_result = {
    "phien": None,
    "xuc_xac_1": -1,
    "xuc_xac_2": -1,
    "xuc_xac_3": -1,
    "tong": -1,
    "ket_qua": None,
    "xac_suat_tai": 0,
    "xac_suat_xiu": 0,
    "trang_thai_cau": "Chưa đủ dữ liệu",
    "loi_khuyen": "Chờ thêm",
    "do_tin_cay": 0,
    "id": "xocdia88-real"
}

# ================== HELPER ==================
def xac_dinh_tai_xiu(tong):
    return "Tài" if tong >= 11 else "Xỉu"

# ================== PHÂN TÍCH CẦU (THẬT) ==================
def phan_tich_10_phien(history):
    if len(history) < 5:
        return 0, 0, "Chưa đủ dữ liệu", "Không nên vào", 0

    last_10 = list(history)[-10:]
    tai = last_10.count("Tài")
    xiu = last_10.count("Xỉu")

    p_tai = round(tai / len(last_10) * 100)
    p_xiu = round(xiu / len(last_10) * 100)

    last_3 = last_10[-3:]
    last_4 = last_10[-4:]

    trang_thai = "Cầu bình thường"
    khuyen = "Theo dõi thêm"
    tin_cay = 50

    # Cầu bệt
    if last_3 == ["Tài"]*3 or last_3 == ["Xỉu"]*3:
        trang_thai = "Cầu bệt (đẹp)"
        khuyen = "Có thể vào tiếp"
        tin_cay = 75

    # Cầu 1-1
    elif last_4 == ["Tài","Xỉu","Tài","Xỉu"] or last_4 == ["Xỉu","Tài","Xỉu","Tài"]:
        trang_thai = "Cầu 1-1 (rất đẹp)"
        khuyen = "Vào được (đánh đảo)"
        tin_cay = 85

    # Cầu xấu
    elif tai >= 8 or xiu >= 8:
        trang_thai = "Cầu xấu / nhiễu"
        khuyen = "KHÔNG NÊN VÀO"
        tin_cay = 20

    # GHI FILE TXT
    with open(FILE_NAME, "w", encoding="utf-8") as f:
        f.write("=== THUẬT TOÁN TÀI XỈU (REAL DATA) ===\n")
        f.write(f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"10 phiên gần nhất: {' - '.join(last_10)}\n")
        f.write(f"Xác suất: Tài {p_tai}% | Xỉu {p_xiu}%\n")
        f.write(f"Trạng thái cầu: {trang_thai}\n")
        f.write(f"Lời khuyên: {khuyen}\n")
        f.write(f"Độ tin cậy: {tin_cay}%\n")

    return p_tai, p_xiu, trang_thai, khuyen, tin_cay

# ================== WEBSOCKET ==================
def on_message(ws, message):
    try:
        data = json.loads(message)
        if "M" not in data:
            return

        for item in data["M"]:
            if item.get("M") != "Md5sessionInfo":
                continue

            info = item["A"][0]
            phien = info.get("SessionID")
            r = info.get("Result", {})

            d1, d2, d3 = r.get("Dice1"), r.get("Dice2"), r.get("Dice3")
            if not all(isinstance(x, int) for x in [d1, d2, d3]):
                return

            tong = d1 + d2 + d3
            ket_qua = xac_dinh_tai_xiu(tong)

            with lock:
                if latest_result["phien"] == phien:
                    return

                if latest_result["ket_qua"]:
                    results_history.append(latest_result["ket_qua"])

                p_tai, p_xiu, trang_thai, khuyen, tin_cay = phan_tich_10_phien(results_history)

                latest_result.update({
                    "phien": phien,
                    "xuc_xac_1": d1,
                    "xuc_xac_2": d2,
                    "xuc_xac_3": d3,
                    "tong": tong,
                    "ket_qua": ket_qua,
                    "xac_suat_tai": p_tai,
                    "xac_suat_xiu": p_xiu,
                    "trang_thai_cau": trang_thai,
                    "loi_khuyen": khuyen,
                    "do_tin_cay": tin_cay
                })

                print(f"[REAL] {phien} | {d1}-{d2}-{d3} | {ket_qua} | {khuyen}")

    except Exception as e:
        print("WS error:", e)

def on_open(ws):
    ws.send(json.dumps({"protocol": "json", "version": 1}) + "\x1e")

def start_ws():
    while True:
        try:
            ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message)
            ws.run_forever(ping_interval=PING_INTERVAL, ping_timeout=5)
        except:
            time.sleep(5)

# ================== API ==================
@app.route("/")
def home():
    return "✅ XocDia88 REAL – Thống kê & soi cầu (10 phiên)"

@app.route("/api/taixiumd5")
def api():
    with lock:
        return jsonify({
            **latest_result,
            "lich_su_10_phien": list(results_history)[-10:]
        })

# ================== MAIN ==================
if __name__ == "__main__":
    print("🚀 Khởi động soi cầu REAL (xocdia88)...")
    threading.Thread(target=start_ws, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
