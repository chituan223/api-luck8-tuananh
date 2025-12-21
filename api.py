from flask import Flask, jsonify
import requests
import statistics
import math
from datetime import datetime

app = Flask(__name__)
PORT = 3000

# ===================== SIÊU THUẬT TOÁN AI (PRO ENSEMBLE) ======================

class TaiXiuSuperAI:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.history = []
        self.totals = []
        # Hệ thống tự học: Khởi tạo trọng số cho 10 chiến thuật mở rộng
        self.weights = [1.0] * 10 
        self.last_predictions = []

    def update_data(self, new_label, new_total):
        # 1. Cơ chế tự học: Kiểm tra kết quả ván trước để cập nhật trọng số uy tín
        if self.last_predictions and self.history:
            actual_last = new_label
            for i, pred in enumerate(self.last_predictions):
                if pred == actual_last:
                    self.weights[i] = min(self.weights[i] + 0.1, 5.0) # Tăng uy tín nếu đoán đúng
                else:
                    self.weights[i] = max(self.weights[i] - 0.1, 0.1) # Giảm uy tín nếu đoán sai

        # 2. Cập nhật dữ liệu mới vào bộ nhớ đệm
        self.history.append(new_label)
        self.totals.append(new_total)
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
            self.totals.pop(0)

    # --- Nhóm Thuật Toán Phân Tích Kỹ Thuật ---
    def ai_rsi_momentum(self):
        if len(self.totals) < 14: return "Tài", 50
        gains = [max(0, self.totals[i] - self.totals[i-1]) for i in range(-13, 0)]
        losses = [max(0, self.totals[i-1] - self.totals[i]) for i in range(-13, 0)]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        rs = avg_gain / (avg_loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        if rsi > 70: return "Xỉu", 88 
        if rsi < 30: return "Tài", 88
        return ("Xỉu" if rsi > 50 else "Tài"), 60

    def ai_bollinger_bands(self):
        if len(self.totals) < 20: return "Xỉu", 50
        sma = statistics.mean(self.totals[-20:])
        std_dev = statistics.stdev(self.totals[-20:])
        upper = sma + (1.8 * std_dev)
        lower = sma - (1.8 * std_dev)
        curr = self.totals[-1]
        if curr > upper: return "Xỉu", 92
        if curr < lower: return "Tài", 92
        return ("Tài" if curr < sma else "Xỉu"), 65

    # --- Nhóm Thuật Toán Xác Suất Chuỗi (Markov) ---
    def ai_markov_depth_2(self):
        if len(self.history) < 15: return "Tài", 50
        pattern = "".join([h[0] for h in self.history[-2:]])
        full_str = "".join([h[0] for h in self.history])
        t_c = full_str.count(pattern + "T")
        x_c = full_str.count(pattern + "X")
        return ("Tài", 85) if t_c > x_c else ("Xỉu", 85)

    def ai_markov_depth_3(self):
        if len(self.history) < 20: return "Xỉu", 50
        pattern = "".join([h[0] for h in self.history[-3:]])
        full_str = "".join([h[0] for h in self.history])
        t_c = full_str.count(pattern + "T")
        x_c = full_str.count(pattern + "X")
        return ("Tài", 90) if t_c > x_c else ("Xỉu", 90)

    # --- Nhóm Thuật Toán Nhận Diện Cầu Thực Tế ---
    def ai_bridge_detector(self):
        """Bắt cầu 1-1, 2-2"""
        if len(self.history) < 4: return "Tài", 50
        h = self.history
        if h[-1] != h[-2] and h[-2] != h[-3]: return ("Tài" if h[-1] == "Xỉu" else "Xỉu"), 85
        if h[-1] == h[-2] and h[-3] == h[-4] and h[-1] != h[-3]: return ("Tài" if h[-1] == "Xỉu" else "Xỉu"), 80
        return h[-1], 55

    def ai_streak_follow(self):
        """Đu bệt khi có dây từ 4 ván trở lên"""
        streak = 1
        for i in range(len(self.history)-1, 0, -1):
            if self.history[i] == self.history[i-1]: streak += 1
            else: break
        if streak >= 4: return self.history[-1], 85
        return ("Xỉu" if self.history[-1] == "Tài" else "Tài"), 60

    # ================= TỔNG HỢP VÀ PHÂN TÍCH CUỐI =================

    def analyze(self):
        if len(self.history) < 15:
            return {"status": "DATA_COLLECTING", "remaining": 15 - len(self.history)}

        # Chạy 6 thuật toán lõi (có thể mở rộng thêm ở đây)
        algo_pool = [
            self.ai_rsi_momentum(),
            self.ai_bollinger_bands(),
            self.ai_markov_depth_2(),
            self.ai_markov_depth_3(),
            self.ai_bridge_detector(),
            self.ai_streak_follow()
        ]

        self.last_predictions = [res[0] for res in algo_pool]
        votes = {"Tài": 0.0, "Xỉu": 0.0}

        for i, (pred, conf) in enumerate(algo_pool):
            # Điểm bầu chọn = (Độ tin cậy thuật toán) * (Trọng số uy tín thực tế)
            votes[pred] += (conf * self.weights[i])

        total_power = votes["Tài"] + votes["Xỉu"]
        decision = "Tài" if votes["Tài"] > votes["Xỉu"] else "Xỉu"
        confidence = (votes[decision] / total_power) * 100

        return {
            "prediction": decision,
            "confidence": f"{round(min(confidence, 98.5), 2)}%",
            "signals": {
                "tai_power": round(votes["Tài"], 1),
                "xiu_power": round(votes["Xỉu"], 1)
            },
            "advice": self._generate_advice(confidence, votes)
        }

    def _generate_advice(self, conf, votes):
        diff = abs(votes["Tài"] - votes["Xỉu"])
        if conf > 85 and diff > 150: return "🔥 TỰ TIN VÀO LỆNH (BIG WIN)"
        if conf > 75: return "✅ CẦU ĐẸP - VÀO ĐỀU TAY"
        if conf > 60: return "⚠️ CẦU NHẸ - ĐÁNH THĂM DÒ"
        return "❌ CẦU LOẠN - NÊN BỎ QUA"

# Khởi tạo Global Bot
bot = TaiXiuSuperAI()

# ======================== API SERVER ========================

@app.route("/api/taixiu/", methods=["GET"])
def taixiu_api():
    try:
        # 1. Fetch dữ liệu từ API sàn
        resp = requests.get("https://1.bot/GetNewLottery/LT_TaixiuMD5", timeout=10)
        data_json = resp.json()
        
        if data_json.get("state") != 1:
            return jsonify({"error": "API_SOURCE_DOWN"}), 503
        
        raw_data = data_json["data"]
        d1, d2, d3 = map(int, raw_data["OpenCode"].split(","))
        total = d1 + d2 + d3
        result = "Tài" if total >= 11 else "Xỉu"

        # 2. Cập nhật dữ liệu vào AI để học và lưu lịch sử
        bot.update_data(result, total)

        # 3. Thực hiện phân tích phiên tiếp theo
        analysis = bot.analyze()

        return jsonify({
            "Phien_hien_tai": raw_data["Expect"],
            "Ket_qua_vua_ra": {
                "Xuc_xac": f"{d1}-{d2}-{d3}",
                "Tong": total,
                "Loai": result
            },
            "Du_doan_AI": analysis,
            "He_thong_tu_hoc": {
                "Do_on_dinh_weights": round(statistics.mean(bot.weights), 2),
                "Phien_da_luu": len(bot.history)
            },
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        })

    except Exception as e:
        return jsonify({"error": "SERVER_ERROR", "details": str(e)}), 500

if __name__ == "__main__":
    print(f"🚀 AI Ensemble System đang chạy tại http://localhost:{PORT}")
    # Tắt debug mode để đảm bảo tính ổn định cho trọng số AI
    app.run(host="0.0.0.0", port=PORT, debug=False)
