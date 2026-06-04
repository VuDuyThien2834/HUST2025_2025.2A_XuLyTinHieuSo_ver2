import numpy as np

def generate_linear_chirp(fs, duration=1.0, f0=100.0, f1=4000.0):
    """
    Tạo tín hiệu Chirp tuyến tính (Linear Frequency Modulated Chirp).
    Tần số thay đổi tuyến tính theo thời gian từ f0 tại t=0 đến f1 tại t=duration.
    
    Công thức toán học:
    x(t) = sin(2 * pi * (f0 * t + 0.5 * k * t^2))
    Trong đó k = (f1 - f0) / duration là tốc độ quét tần.
    
    Trả về:
    - t: Mảng thời gian (1D array)
    - x: Mảng tín hiệu Chirp (1D array)
    - if_true: Tần số tức thời thực tế (ground truth) tại mỗi điểm thời gian (dùng để so sánh độ chính xác)
    """
    t = np.arange(0, int(fs * duration)) / fs
    k = (f1 - f0) / duration
    
    # Tính pha toàn cục của tín hiệu Chirp tuyến tính
    phase = 2 * np.pi * (f0 * t + 0.5 * k * (t ** 2))
    x = np.sin(phase)
    
    # Tần số tức thời thực tế: f_i(t) = f0 + k*t
    if_true = f0 + k * t
    
    return t, x, if_true

def generate_gaussian_pulse(fs, sigma=0.05, duration=1.0):
    """
    Tạo một xung Gaussian chuẩn hóa năng lượng phục vụ cho thí nghiệm Nguyên lý bất định.
    Tâm của xung được đặt chính giữa khoảng thời gian (t = duration / 2).
    
    Công thức toán học:
    x(t) = (1 / (pi * sigma^2))^(1/4) * exp(-(t - t_center)^2 / (2 * sigma^2))
    
    Trả về:
    - t: Mảng thời gian
    - x: Mảng tín hiệu xung Gaussian
    """
    t = np.arange(0, int(fs * duration)) / fs
    t_center = duration / 2.0
    
    # Hệ số chuẩn hóa biên độ để đảm bảo năng lượng tín hiệu tích phân bằng 1
    norm_factor = (1.0 / (np.pi * (sigma ** 2))) ** 0.25
    
    # Tính toán dạng sóng xung
    x = norm_factor * np.exp(-((t - t_center) ** 2) / (2.0 * (sigma ** 2)))
    
    return t, x

def generate_bat_call(fs, duration=0.05):
    """
    Mô phỏng tiếng kêu định vị của dơi (Bat echolocation call).
    Đây là một tín hiệu FM chirp phi tuyến (Hyperbolic/Exponential Chirp) đặc trưng,
    tần số quét cực nhanh từ khoảng 80 kHz xuống 20 kHz trong thời gian rất ngắn (~수 chục ms).
    Ở đây cấu hình tỷ lệ tương đương về dải tần nghe được (audio range) để tiện phân tích.
    
    Tần số quét phi tuyến từ 6000 Hz xuống 1000 Hz theo dạng hàm mũ.
    
    Trả về:
    - t: Mảng thời gian
    - x: Mảng tín hiệu tiếng dơi mô phỏng
    - if_true: Tần số tức thời thực tế (ground truth) dạng hàm mũ
    """
    t = np.arange(0, int(fs * duration)) / fs
    f0 = 6000.0  # Tần số bắt đầu (cao)
    f1 = 1000.0  # Tần số kết thúc (thấp)
    
    # Tính toán hệ số suy giảm hàm mũ cho tần số
    # f(t) = f0 * (f1 / f0)^(t / duration)
    b = np.log(f1 / f0) / duration
    
    # Pha toàn cục là tích phân của tần số: Phase(t) = 2 * pi * \int f(t) dt
    phase = 2 * np.pi * f0 * (np.exp(b * t) - 1.0) / b
    x = np.sin(phase)
    
    # Tần số tức thời thực tế tại mỗi điểm t
    if_true = f0 * np.exp(b * t)
    
    return t, x, if_true