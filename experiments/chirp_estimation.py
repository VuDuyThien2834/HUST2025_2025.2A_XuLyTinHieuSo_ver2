import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from signal_generator.synthetic import generate_linear_chirp

def run_chirp_estimation_experiment():
    fs = 8000
    duration = 1.0
    f0, f1 = 200.0, 3000.0
    
    # 1. Sinh tín hiệu Chirp và lấy đường chuẩn tần số tức thời gốc
    t, x, if_true = generate_linear_chirp(fs, duration, f0, f1)
    
    # 2. Tạo tín hiệu giải tích qua biến đổi Hilbert
    z = hilbert(x)
    
    # 3. Trích xuất pha tức thời (Phép giải bọc pha - Unwrapping bắt buộc)
    phase_instantaneous = np.unwrap(np.angle(z))
    
    # 4. Tính toán tần số tức thời bằng sai phân đạo hàm: f = (1/2pi) * d(phi)/dt
    # Do phép sai phân np.diff làm mảng hụt đi 1 phần tử, ta tiến hành bù phần tử cuối
    if_estimated = np.diff(phase_instantaneous) / (2.0 * np.pi) * fs
    if_estimated = np.append(if_estimated, if_estimated[-1]) 
    
    # 5. Tính toán sai số trung bình tuyệt đối (MAE) để đưa vào bảng thống kê báo cáo
    mae = np.mean(np.abs(if_true - if_estimated))
    print(f"📊 Kết quả kiểm toán chất lượng ước lượng tần số:")
    print(f"   - Sai số trung bình tuyệt đối (MAE): {mae:.4f} Hz")
    
    # Vẽ đồ thị so sánh giữa Đường chuẩn và Đường ước lượng thuật toán
    plt.figure(figsize=(10, 5))
    plt.plot(t, if_true, label='Tần số tức thời Chuẩn (Ground Truth)', color='black', linewidth=2)
    plt.plot(t, if_estimated, '--', label='Tần số ước lượng (Hilbert Transform)', color='cyan', linewidth=1.5)
    plt.title("So sánh Tần số tức thời thực tế và Tần số trích xuất từ Tín hiệu Giải tích")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Tần số (Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("data/saved_plots/chirp_frequency_estimation.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_chirp_estimation_experiment()