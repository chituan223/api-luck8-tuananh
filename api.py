from flask import Flask, jsonify
import requests
import threading
import time
import statistics
import re
import json

# --- CÁC HẰNG SỐ VÀ DỮ LIỆU TOÀN CỤC ---
API_SOURCE = "https://1.bot/GetNewLottery/LT_TaixiuMD5"
MAX_HISTORY = 50        # Chiều dài lịch sử tối đa

pattern_history = []    # Lưu trữ kết quả Tài/Xỉu (Tài/Xỉu)
sums_history = []       # Lưu trữ tổng điểm xúc xắc (3-18)
# Bộ theo dõi hiệu suất: (wins, total_rounds)
algo_performance = [(0, 0) for _ in range(20)] 

last_expect = None
current_data = None     # Dữ liệu phiên mới nhất


# --- CÁC HÀM TIỆN ÍCH CƠ BẢN ---

def confidence_to_scores(prediction, confidence):
    """Chuyển đổi dự đoán và độ tin cậy thành điểm số Tài/Xỉu (Base Score)."""
    score = {}
    # Giới hạn confidence để đảm bảo luôn có sự phân biệt rõ ràng
    confidence = min(100.0, max(50.0, confidence)) 
    if prediction == "Tài":
        score['tai'] = confidence
        score['xiu'] = 100 - confidence
    else: # Xỉu
        score['xiu'] = confidence
        score['tai'] = 100 - confidence
    return score

def get_taixiu(total):
    """Xác định Tài/Xỉu dựa trên tổng điểm xúc xắc."""
    if 3 <= total <= 10:
        return "Xỉu"
    if 11 <= total <= 18:
        return "Tài"
    return "Lỗi"

def update_history(result, total):
    """Cập nhật lịch sử kết quả và tổng điểm."""
    global pattern_history, sums_history
    pattern_history.append(result)
    sums_history.append(total)
    
    # Giới hạn chiều dài lịch sử
    if len(pattern_history) > MAX_HISTORY:
        pattern_history.pop(0)
        sums_history.pop(0)

def safe_mean(data):
    """Hàm an toàn để tính mean."""
    if not data:
        return 10.5 # Giá trị trung lập
    return statistics.mean(data)


# --- PHẦN HỆ THỐNG DỰ ĐOÁN (20 THUẬT TOÁN ĐÃ ĐƯỢC ẨN CHI TIẾT) ---

# Tên các thuật toán để theo dõi hiệu suất
ALGO_NAMES = [
    "AI-1 Frequency (Tần suất)", "AI-2 Parity Chain (Chuỗi chẵn lẻ)", "AI-3 Moving Avg (TB trượt)",
    "AI-4 Streak Detector (Phát hiện chuỗi)", "AI-5 Alternating Pattern (Mô hình xen kẽ)",
    "AI-6 Total Variability (Biến động tổng)", "AI-7 Short Cycle (Chu kỳ ngắn)",
    "AI-8 Even Bias Long (Thiên vị chẵn dài)", "AI-9 Median Check (Kiểm tra Trung vị)",
    "AI-10 Trend Slope (Độ dốc xu hướng)", "AI-11 Weighted Vote (Bỏ phiếu trọng số)",
    "AI-12 Recent Trend (Xu hướng gần đây)", "AI-13 Balance (Cân bằng T/X)",
    "AI-14 Gradient (Độ biến thiên)", "AI-15 Stability (Ổn định)",
    "AI-16 Flip After Loss (Đảo chiều sau chuỗi)", "AI-17 Recent Variance (Phương sai gần)",
    "AI-18 Sequence (Dãy lặp)", "AI-19 Long Term Mean (TB dài hạn)", "AI-20 Adaptive (Thích ứng)"
]

# 20 HÀM AI (CHỈ SỬ DỤNG LÝ THUYẾT TẬP HỢP, KHÔNG RANDOM)
# Giữ nguyên logic cơ bản của 20 AI con, đảm bảo tính deterministic.

def _ai_1_frequency(lich_su, tong_diem):
    if len(lich_su) < 6: return confidence_to_scores("Tài", 65.2)
    cua_so = lich_su[-6:]; t = cua_so.count("Tài"); x = cua_so.count("Xỉu")
    if t >= x + 3: return confidence_to_scores("Xỉu", 90.0)
    if x >= t + 3: return confidence_to_scores("Tài", 90.0)
    if t == x + 2: return confidence_to_scores("Tài", 78.0)
    if x == t + 2: return confidence_to_scores("Xỉu", 78.0)
    return confidence_to_scores(lich_su[-1], 70.0)

def _ai_2_parity_chain(lich_su, tong_diem):
    if len(tong_diem) < 5: return confidence_to_scores("Tài", 66.7)
    nam_cuoi = tong_diem[-5:]; so_chan = sum(1 for t in nam_cuoi if t % 2 == 0)
    if so_chan >= 4: return confidence_to_scores("Xỉu", 92.0)
    if so_chan <= 1: return confidence_to_scores("Tài", 92.0)
    return confidence_to_scores("Tài" if tong_diem[-1] >= 11 else "Xỉu", 75.0)

def _ai_3_moving_avg(lich_su, tong_diem):
    if len(tong_diem) < 4: return confidence_to_scores("Tài", 65.8)
    trung_binh4 = safe_mean(tong_diem[-4:])
    if trung_binh4 > 11.5: return confidence_to_scores("Tài", 88.0)
    if trung_binh4 < 9.5: return confidence_to_scores("Xỉu", 88.0)
    return confidence_to_scores(lich_su[-1], 72.1)

def _ai_4_streak_detector(lich_su, tong_diem):
    if len(lich_su) < 5: return confidence_to_scores("Tài", 64.3)
    ket_qua_cuoi = lich_su[-1]; chuoi = 1
    for i in range(len(lich_su) - 2, -1, -1):
        if lich_su[i] == ket_qua_cuoi: chuoi += 1
        else: break
    if chuoi >= 5: return confidence_to_scores("Xỉu" if ket_qua_cuoi == "Tài" else "Tài", 93.0)
    return confidence_to_scores(ket_qua_cuoi, 70.0)

def _ai_5_alternating_pattern(lich_su, tong_diem):
    if len(lich_su) < 6: return confidence_to_scores("Tài", 66.2)
    chuoi = "".join("T" if h == "Tài" else "X" for h in lich_su[-6:])
    if chuoi.endswith("TXTX") or chuoi.endswith("XTXT"):
        du_doan_tiep = "Tài" if chuoi[-1] == "X" else "Xỉu"
        return confidence_to_scores(du_doan_tiep, 90.0)
    return confidence_to_scores(lich_su[-1], 68.9)

def _ai_6_total_variability(lich_su, tong_diem):
    if len(tong_diem) < 5: return confidence_to_scores("Tài", 67.0)
    cua_so = tong_diem[-5:]; trung_binh = safe_mean(cua_so); bien_dong = max(cua_so) - min(cua_so)
    if trung_binh >= 11 and bien_dong <= 1: return confidence_to_scores("Tài", 91.0)
    if trung_binh <= 10 and bien_dong <= 1: return confidence_to_scores("Xỉu", 91.0)
    if trung_binh >= 11 and bien_dong <= 2: return confidence_to_scores("Tài", 80.0)
    if trung_binh <= 10 and bien_dong <= 2: return confidence_to_scores("Xỉu", 80.0)
    return confidence_to_scores(lich_su[-1], 70.0)

def _ai_7_short_cycle(lich_su, tong_diem):
    if len(lich_su) < 3: return confidence_to_scores("Tài", 61.7)
    duoi = lich_su[-3:]
    if duoi[0] == duoi[2] and duoi[0] != duoi[1]:
        du_doan_tiep = duoi[1]
        return confidence_to_scores(du_doan_tiep, 88.9)
    return confidence_to_scores(lich_su[-1], 70.3)

def _ai_8_even_bias_long(lich_su, tong_diem):
    if len(tong_diem) < 8: return confidence_to_scores("Tài", 64.6)
    tam_cuoi = tong_diem[-8:]; so_chan = sum(1 for t in tam_cuoi if t % 2 == 0)
    if so_chan >= 7: return confidence_to_scores("Xỉu", 93.0)
    if so_chan <= 1: return confidence_to_scores("Tài", 93.0)
    return confidence_to_scores("Tài" if tong_diem[-1] >= 11 else "Xỉu", 71.5)

def _ai_9_median_check(lich_su, tong_diem):
    if len(tong_diem) < 5: return confidence_to_scores("Tài", 65.1)
    try: trung_vi = statistics.median(tong_diem[-5:])
    except: return confidence_to_scores("Tài", 65.1)
    if trung_vi > 11.5: return confidence_to_scores("Tài", 88.0)
    if trung_vi < 9.5: return confidence_to_scores("Xỉu", 88.0)
    return confidence_to_scores(lich_su[-1], 70.0)

def _ai_10_trend_slope(lich_su, tong_diem):
    if len(tong_diem) < 5: return confidence_to_scores("Tài", 63.7)
    do_doc = (tong_diem[-1] - tong_diem[-5]) / 4
    if do_doc >= 0.8: return confidence_to_scores("Tài", 90.0)
    if do_doc <= -0.8: return confidence_to_scores("Xỉu", 90.0)
    return confidence_to_scores("Tài" if tong_diem[-1] >= 11 else "Xỉu", 72.2)

def _ai_11_weighted_vote(lich_su, tong_diem):
    if len(lich_su) < 6 or len(tong_diem) < 6: return confidence_to_scores("Tài", 66.4)
    dem_t = lich_su[-6:].count("Tài"); tb6 = safe_mean(tong_diem[-6:])
    chan_le = sum(1 for t in tong_diem[-6:] if t % 2 == 0); diem = 0
    if dem_t >= 4: diem += 1
    if tb6 >= 11.5: diem += 1
    if chan_le <= 2: diem += 1
    if dem_t <= 2: diem -= 1
    if tb6 <= 9.5: diem -= 1
    if chan_le >= 4: diem -= 1
    if diem >= 2: return confidence_to_scores("Tài", 88.0)
    if diem <= -2: return confidence_to_scores("Xỉu", 88.0)
    return confidence_to_scores(lich_su[-1], 75.0)

def _ai_12_recent_trend(lich_su, tong_diem):
    if len(lich_su) < 3: return confidence_to_scores("Tài", 62.3)
    xu_huong = lich_su[-2:]
    if xu_huong[0] == xu_huong[1]: return confidence_to_scores(xu_huong[0], 85.0)
    return confidence_to_scores(lich_su[-1], 70.0)

def _ai_13_balance(lich_su, tong_diem):
    t = lich_su.count("Tài"); x = lich_su.count("Xỉu")
    if abs(t - x) >= 7: return confidence_to_scores("Xỉu" if t > x else "Tài", 85.0)
    return confidence_to_scores(lich_su[-1], 71.6)

def _ai_14_gradient(lich_su, tong_diem):
    if len(tong_diem) < 4: return confidence_to_scores("Tài", 63.4)
    do_phan_cuc = tong_diem[-1] - tong_diem[-4]
    if do_phan_cuc > 2.0: return confidence_to_scores("Tài", 88.0)
    if do_phan_cuc < -2.0: return confidence_to_scores("Xỉu", 88.0)
    return confidence_to_scores(lich_su[-1], 74.0)

def _ai_15_stability(lich_su, tong_diem):
    if len(tong_diem) < 5: return confidence_to_scores("Tài", 64.5)
    do_lech = max(tong_diem[-5:]) - min(tong_diem[-5:])
    if do_lech <= 2 and tong_diem[-1] >= 11: return confidence_to_scores("Xỉu", 85.0)
    if do_lech <= 2 and tong_diem[-1] <= 10: return confidence_to_scores("Tài", 85.0)
    return confidence_to_scores("Tài", 70.0)

def _ai_16_flip_after_loss(lich_su, tong_diem):
    if len(lich_su) < 3: return confidence_to_scores("Tài", 72.6)
    ba_cuoi = lich_su[-3:]
    if ba_cuoi[0] == ba_cuoi[1] == ba_cuoi[2]: return confidence_to_scores("Xỉu" if ba_cuoi[0] == "Tài" else "Tài", 85.0)
    return confidence_to_scores(lich_su[-1], 72.6)

def _ai_17_recent_variance(lich_su, tong_diem):
    if len(tong_diem) < 5: return confidence_to_scores("Tài", 66.1)
    phuong_sai = max(tong_diem[-5:]) - min(tong_diem[-5:])
    if phuong_sai > 5: return confidence_to_scores("Tài", 80.0)
    if phuong_sai <= 2: return confidence_to_scores("Xỉu", 80.0)
    return confidence_to_scores(lich_su[-1], 70.0)

def _ai_18_sequence(lich_su, tong_diem):
    if len(lich_su) < 5: return confidence_to_scores("Tài", 64.9)
    chuoi = "".join("T" if h == "Tài" else "X" for h in lich_su[-5:])
    if chuoi == "TTTTT": return confidence_to_scores("Xỉu", 91.0)
    if chuoi == "XXXXX": return confidence_to_scores("Tài", 91.0)
    return confidence_to_scores(lich_su[-1], 70.9)

def _ai_19_long_term_mean(lich_su, tong_diem):
    if len(tong_diem) < 10: return confidence_to_scores("Tài", 65.7)
    tb10 = safe_mean(tong_diem[-10:])
    if tb10 > 11.2: return confidence_to_scores("Tài", 86.0)
    if tb10 < 9.8: return confidence_to_scores("Xỉu", 86.0)
    return confidence_to_scores(lich_su[-1], 71.3)

def _ai_20_adaptive(lich_su, tong_diem):
    if len(lich_su) < 8: return confidence_to_scores("Tài", 66.5)
    dem_t = lich_su[-8:].count("Tài"); ty_le = dem_t / 8
    if ty_le >= 0.875: return confidence_to_scores("Xỉu", 92.0)
    if ty_le <= 0.125: return confidence_to_scores("Tài", 92.0)
    return confidence_to_scores(lich_su[-1], 72.4)

ALGOS_SUPER_STANDARD = [
    _ai_1_frequency, _ai_2_parity_chain, _ai_3_moving_avg, _ai_4_streak_detector,
    _ai_5_alternating_pattern, _ai_6_total_variability, _ai_7_short_cycle,
    _ai_8_even_bias_long, _ai_9_median_check, _ai_10_trend_slope,
    _ai_11_weighted_vote, _ai_12_recent_trend, _ai_13_balance, _ai_14_gradient,
    _ai_15_stability, _ai_16_flip_after_loss, _ai_17_recent_variance,
    _ai_18_sequence, _ai_19_long_term_mean, _ai_20_adaptive
]


def Hybrid_SuperStandard(history, sums, performance_data):
    """
    Hàm kết hợp 20 thuật toán bằng cơ chế Weighted Score Voting (Bỏ phiếu trọng số)
    dựa trên HIỆU SUẤT LỊCH SỬ của từng AI.
    """
    if len(history) < 3:
        # Trả về kết quả trung lập khi chưa đủ dữ liệu lịch sử
        return "Tài", 50.0 
        
    total_weighted_score = {'tai': 0.0, 'xiu': 0.0}
    active_algos = 0
    
    # Lấy tổng số vòng đã theo dõi để xác định AI nào đã "khởi động"
    total_rounds_tracked = performance_data[0][1] 

    # Chạy 20 thuật toán cơ bản và tính tổng điểm tin cậy có trọng số
    for i, algo in enumerate(ALGOS_SUPER_STANDARD):
        wins, total = performance_data[i]
        
        # 1. Xác định trọng số (Weight)
        if total_rounds_tracked == 0 or total == 0:
             # Nếu chưa có lịch sử, gán trọng số trung lập (50%)
            weight = 0.5 
        else:
            # Trọng số = Tỷ lệ thắng thực tế
            win_rate = wins / total
            # Giới hạn trọng số từ 0.35 đến 0.95 để tránh quá phân cực
            weight = max(0.35, min(0.95, win_rate))

        try:
            # 2. Lấy điểm tin cậy cơ bản từ AI
            result_scores = algo(history, sums)
            
            # 3. Tính điểm có trọng số (Weighted Score)
            # Điểm cuối = (Điểm tin cậy Tài/Xỉu) * Trọng số
            
            # Nếu AI dự đoán Tài
            if result_scores['tai'] > result_scores['xiu']:
                # Cộng điểm tin cậy vào Tài, và (100 - tin cậy) vào Xỉu, nhân với trọng số
                total_weighted_score['tai'] += result_scores['tai'] * weight
                total_weighted_score['xiu'] += result_scores['xiu'] * weight
            # Nếu AI dự đoán Xỉu
            else: 
                # Cộng điểm tin cậy vào Xỉu, và (100 - tin cậy) vào Tài, nhân với trọng số
                total_weighted_score['xiu'] += result_scores['xiu'] * weight
                total_weighted_score['tai'] += result_scores['tai'] * weight
                
            active_algos += 1
            
        except Exception:
            continue

    # Kết luận: Cửa nào có tổng điểm tin cậy có trọng số cao hơn sẽ được chọn
    final_prediction = "Tài" if total_weighted_score['tai'] >= total_weighted_score['xiu'] else "Xỉu"
    
    # Tính độ tin cậy trung bình
    total_confidence = total_weighted_score['tai'] + total_weighted_score['xiu']
    
    # Tính phần trăm độ tin cậy cuối cùng dựa trên tổng điểm có trọng số
    confidence_percentage = (max(total_weighted_score['tai'], total_weighted_score['xiu']) / total_confidence) * 100 if total_confidence > 0 else 50.0

    # Nếu chưa có đủ vòng theo dõi, giữ confidence ở mức trung tính hơn
    if total_rounds_tracked < 10:
        confidence_percentage = max(55.0, min(80.0, confidence_percentage))
    
    return final_prediction, confidence_percentage


def track_individual_performance(current_result):
    """Theo dõi hiệu suất của từng AI đơn lẻ."""
    global pattern_history, sums_history, algo_performance
    
    # Lịch sử hiện tại là lịch sử N-1 (dùng để dự đoán)
    # Vì hàm này được gọi TRƯỚC khi update_history, lịch sử đã có là N-1, đủ để dự đoán cho kết quả N
    history_n_minus_1 = pattern_history
    sums_n_minus_1 = sums_history
    
    if len(history_n_minus_1) < 1: # Cần ít nhất 1 kết quả để dự đoán kết quả tiếp theo
        return

    # Chỉ theo dõi các AI đã có ít nhất 1 lần chạy hợp lệ
    start_index = 0
    # Ví dụ: Nếu tổng vòng theo dõi > 10, bắt đầu theo dõi tất cả 20 AI
    if algo_performance[0][1] > 10: 
        start_index = 0
    else:
        # Nếu chưa đủ dữ liệu, chỉ theo dõi các AI đơn giản (ví dụ: 5 AI đầu)
        pass 


    for i, algo in enumerate(ALGOS_SUPER_STANDARD):
        wins, total = algo_performance[i]
        
        # Nếu chưa đủ lịch sử cho AI này (ví dụ: AI-19 cần 10 vòng), thì không track
        if len(history_n_minus_1) < 10 and i >= 18: continue
        
        ai_pred = "Lỗi"
        try:
            result_scores = algo(history_n_minus_1, sums_n_minus_1)
            # Dự đoán (Predict) - Chọn cửa có điểm cao hơn
            ai_pred = "Tài" if result_scores['tai'] >= result_scores['xiu'] else "Xỉu"
        except Exception:
            ai_pred = "Lỗi"
            
        # Cập nhật hiệu suất
        total += 1
        if ai_pred == current_result:
            wins += 1
            
        algo_performance[i] = (wins, total)


# --- THUẬT TOÁN ĐỌC DỮ LIỆU NỀN VÀ XỬ LÝ (BACKGROUND LOOP) ---
app = Flask(__name__)

def background_loop():
    global last_expect, current_data, algo_performance

    max_retries = 5
    initial_delay = 2 

    while True:
        data = None
        # Lấy dữ liệu API với Exponential Backoff
        for attempt in range(max_retries):
            try:
                # API_SOURCE đang là domain dummy, thực tế cần thay bằng domain thật
                res = requests.get(API_SOURCE, timeout=5) 
                res.raise_for_status()
                data = res.json()
                break
            except Exception as e:
                # Nếu không phải lỗi HTTP hoặc mạng, log lỗi và thử lại
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    print(f"Lỗi kết nối API sau {max_retries} lần thử: {e}. Đang đợi 30s...")
                    time.sleep(30) 
                    break
        
        if data is None or data.get("state") != 1:
            time.sleep(2)
            continue
            
        try:
            info = data["data"]
            # Đảm bảo có OpenCode và là phiên mới
            if not info.get("OpenCode") or info["Expect"] == last_expect:
                time.sleep(1)
                continue

            # 1. Tách xúc xắc và tính toán kết quả thực tế (Phiên N vừa ra)
            xuc_xac_matches = re.findall(r'\d+', info["OpenCode"])
            if len(xuc_xac_matches) < 3:
                # Dữ liệu OpenCode không hợp lệ
                time.sleep(2)
                continue
                
            x1, x2, x3 = map(int, xuc_xac_matches)
            tong = x1 + x2 + x3
            ketqua = get_taixiu(tong)

            # 2. Track hiệu suất AI (Dựa trên lịch sử N-1 để so với kết quả N vừa ra)
            if last_expect is not None:
                track_individual_performance(ketqua) 

            # 3. Cập nhật lịch sử toàn cục (Thêm kết quả N)
            update_history(ketqua, tong)

            # 4. Dự đoán cho phiên kế tiếp (N+1)
            # Truyền performance_data vào để tính trọng số
            du_doan_tiep_theo, confidence_percentage = Hybrid_SuperStandard(
                pattern_history, 
                sums_history, 
                algo_performance
            )

            # 5. Tổng hợp hiệu suất các AI (Chỉ dùng để hiển thị)
            performance_summary = []
            for i, (wins, total) in enumerate(algo_performance):
                win_rate = (wins / total) * 100 if total > 0 else 0
                performance_summary.append({
                    "name": ALGO_NAMES[i],
                    "win_rate": round(win_rate, 2),
                    "wins": wins,
                    "total": total
                })
            # Sắp xếp các AI theo tỷ lệ thắng để người dùng dễ theo dõi
            performance_summary.sort(key=lambda x: x['win_rate'], reverse=True)

            # 6. Cập nhật dữ liệu trả về API
            
            # Chỉ lấy 10 kết quả gần nhất cho phản hồi API
            recent_history_detailed = []
            for i in range(max(0, len(pattern_history) - 10), len(pattern_history)):
                recent_history_detailed.append(pattern_history[i])


            current_data = {
                "phien": info["Expect"],
                "xuc_xac_1": x1, "xuc_xac_2": x2, "xuc_xac_3": x3,
                "tong_diem": tong,
                "ket_qua": ketqua, # Kết quả của phiên vừa ra
                "du_doan_tiep_theo": du_doan_tiep_theo, # Dự đoán cho phiên kế tiếp (ĐÃ ĐƯỢC ĐIỀU CHỈNH TRỌNG SỐ)
                "do_tin_cay": round(confidence_percentage, 2),
                "lich_su_gan_nhat": recent_history_detailed,
                "hieu_suat_ai": performance_summary[:5] # Chỉ trả về 5 AI tốt nhất
            }
            last_expect = info["Expect"]

        except Exception as e:
            # Lỗi xử lý dữ liệu JSON hoặc logic
            print(f"Lỗi xử lý logic: {e}")

        time.sleep(2)  # Delay cho loop


# --- API FLASK ---
@app.route("/api/taixiumd5", methods=["GET"])
def api_taixiu():
    """Trả về kết quả phiên mới nhất và dự đoán cho phiên kế tiếp."""
    if current_data is None:
        return jsonify({"state": 0, "msg": "Đang lấy dữ liệu và khởi tạo AI. Vui lòng đợi trong giây lát..."}), 200

    return jsonify({"state": 10, "data": current_data}), 200


# --- KHỞI ĐỘNG SERVER ---
if __name__ == "__main__":
    thread = threading.Thread(target=background_loop)
    thread.daemon = True
    thread.start()

    # Chạy Flask ở port 10000
    app.run(host="0.0.0.0", port=10000)
