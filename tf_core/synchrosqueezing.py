import numpy as np
from tf_core.stft_custom import stft_custom

def synchrosqueezed_stft(x, fs, window_type='hamming', n_per_seg=512, hop_size=256, n_fft=512, threshold=1e-5):
    """
    Triển khai Thuật toán Nén đồng bộ dựa trên STFT (Synchrosqueezed STFT - FSST).
    
    Tham số:
    - x: Mảng tín hiệu đầu vào 1D.
    - fs: Tần số lấy mẫu (Hz).
    - window_type: Loại cửa sổ sử dụng.
    - n_per_seg: Chiều dài cửa sổ (N).
    - hop_size: Bước nhảy khung (H).
    - n_fft: Kích thước phân tích FFT.
    - threshold: Ngưỡng năng lượng (gamma) để loại bỏ nhiễu nền khi tính toán pha.
    
    Trả về:
    - f: Trục tần số vật lý tuyến tính ban đầu (Trục Y của STFT).
    - t: Trục thời gian vật lý (Trục X).
    - sst_matrix: Ma trận kết quả sau nén đồng bộ [Số vạch tần số x Số điểm thời gian].
    """
    # 1. Tính toán ma trận STFT tiêu chuẩn của tín hiệu x
    f, t, Zxx = stft_custom(x, fs, window_type=window_type, n_per_seg=n_per_seg, hop_size=hop_size, n_fft=n_fft)
    
    n_freqs, n_frames = Zxx.shape
    
    # 2. Tính toán đạo hàm của STFT theo thời gian dV/dt
    # Để tính đạo hàm chính xác mà không dùng hàm scipy, ta sử dụng sai phân trung tâm (Central Difference)
    # trên ma trận STFT theo trục thời gian (axis=1)
    dZxx_dt = np.zeros_like(Zxx, dtype=complex)
    
    # Sai phân tiến tại biên đầu, sai phân lùi tại biên cuối
    dt_step = hop_size / fs
    dZxx_dt[:, 0] = (Zxx[:, 1] - Zxx[:, 0]) / dt_step
    dZxx_dt[:, -1] = (Zxx[:, -1] - Zxx[:, -2]) / dt_step
    
    # Sai phân trung tâm cho các khung bên trong
    dZxx_dt[:, 1:-1] = (Zxx[:, 2:] - Zxx[:, :-2]) / (2.0 * dt_step)
    
    # 3. Tính toán toán tử Tần số tức thời (Instantaneous Frequency Estimator - omega)
    # Công thức: omega(t, eta) = round( Im( (dZxx/dt) / Zxx ) / (2 * pi) )
    # Khởi tạo ma trận chứa chỉ số tần số tái phân bổ
    if_indices = np.zeros_like(Zxx, dtype=int)
    
    for m in range(n_frames):
        for k in range(n_freqs):
            v_val = Zxx[k, m]
            # Chỉ tính toán tại các ô tọa độ có năng lượng vượt ngưỡng threshold để tránh chia cho 0 hoặc nhiễu số học
            if np.abs(v_val) > threshold:
                # Ước lượng tần số góc tức thời (radians/second)
                omega_val = np.imag(dZxx_dt[k, m] / v_val)
                # Chuyển đổi từ tần số góc sang Hertz (Hz)
                freq_hz = omega_val / (2.0 * np.pi)
                
                # Ánh xạ tần số vật lý vừa tìm được sang chỉ số ô (index) gần nhất trên lưới tần số f
                if 0 <= freq_hz <= fs / 2.0:
                    if_indices[k, m] = np.argmin(np.abs(f - freq_hz))
                else:
                    if_indices[k, m] = -1 # Nằm ngoài dải Nyquist khả thi
            else:
                if_indices[k, m] = -1
                
    # 4. Thực hiện bước Nén đồng bộ (Squeezing/Reassignment)
    # Gom tụ năng lượng bị nhòe về đúng chỉ số tần số tức thời thực tế
    sst_matrix = np.zeros_like(Zxx, dtype=complex)
    
    for m in range(n_frames):
        for k in range(n_freqs):
            target_freq_idx = if_indices[k, m]
            # Nếu chỉ số hợp lệ, thực hiện cộng dồn tích lũy năng lượng phổ vào tọa độ mới
            if target_freq_idx != -1:
                sst_matrix[target_freq_idx, m] += Zxx[k, m]
                
    return f, t, sst_matrix

def get_sst_db(sst_matrix):
    """
    Chuyển đổi ma trận SST sang thang đo Decibel (dB) phục vụ hiển thị.
    """
    magnitude = np.abs(sst_matrix)
    sst_db = 20 * np.log10(magnitude + 1e-10)
    return sst_db