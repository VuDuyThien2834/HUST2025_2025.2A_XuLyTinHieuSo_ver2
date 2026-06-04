import numpy as np

def generate_window(window_type, n_per_seg):
    """
    Tự lập trình các hàm cửa sổ từ đầu mà không dùng scipy.signal.windows.
    """
    n = np.arange(n_per_seg)
    if window_type == 'rectangular':
        return np.ones(n_per_seg)
    elif window_type == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (n_per_seg - 1))
    elif window_type == 'hann':
        return 0.5 * (1 - np.cos(2 * np.pi * n / (n_per_seg - 1)))
    elif window_type == 'blackman':
        return 0.42 - 0.5 * np.cos(2 * np.pi * n / (n_per_seg - 1)) + 0.08 * np.cos(4 * np.pi * n / (n_per_seg - 1))
    elif window_type == 'bartlett':
        return 1.0 - np.abs((n - (n_per_seg - 1) / 2) / ((n_per_seg - 1) / 2))
    elif window_type == 'kaiser':
        # Mặc định beta = 5.0 nếu người dùng chọn Kaiser đơn giản
        # Sử dụng khai triển chuỗi Taylor cho hàm Bessel loại I bậc 0 (I_0)
        def io(x, exact=False):
            return np.i0(x) # Có thể dùng np.i0 của numpy để tính nhanh hàm Bessel
        beta = 5.0
        return io(beta * np.sqrt(1 - ((n - (n_per_seg - 1) / 2) / ((n_per_seg - 1) / 2))**2)) / io(beta)
    else:
        raise ValueError(f"Không hỗ trợ loại cửa sổ: {window_type}")

def stft_custom(x, fs, window_type='hamming', n_per_seg=512, hop_size=256, n_fft=512):
    """
    Tính toán Biến đổi Fourier thời gian ngắn (STFT) từ đầu.
    
    Tham số:
    - x: Mảng tín hiệu đầu vào (1D array)
    - fs: Tần số lấy mẫu (Hz)
    - window_type: Loại cửa sổ ('rectangular', 'hamming', 'hann', 'blackman', 'bartlett', 'kaiser')
    - n_per_seg: Chiều dài của một khung/cửa sổ (N)
    - hop_size: Bước nhảy giữa các khung (Hop size / H)
    - n_fft: Số điểm FFT (n_fft >= n_per_seg)
    
    Trả về:
    - f: Mảng chứa các vạch tần số (Trục Y)
    - t: Mảng chứa các điểm thời gian (Trục X)
    - Zxx: Ma trận kết quả STFT phức (2D array, kích thước: [Số vạch tần số, Số khung thời gian])
    """
    # 1. Khởi tạo hàm cửa sổ và chuẩn hóa
    win = generate_window(window_type, n_per_seg)
    
    # 2. Tính toán kích thước ma trận đầu ra
    x_len = len(x)
    # Tính số lượng khung thời gian (Frames)
    n_frames = int(np.floor((x_len - n_per_seg) / hop_size)) + 1
    
    # Số lượng vạch tần số cho phổ một phía (One-sided FFT đối với tín hiệu thực)
    n_freqs = n_fft // 2 + 1
    
    # Khởi tạo ma trận trống để chứa kết quả phức
    Zxx = np.zeros((n_freqs, n_frames), dtype=complex)
    
    # 3. Vòng lặp chia khung, nhân cửa sổ và tính FFT
    for m in range(n_frames):
        start_idx = m * hop_size
        end_idx = start_idx + n_per_seg
        
        # Cắt một đoạn tín hiệu (khung)
        x_frame = x[start_idx:end_idx]
        
        # Nhân tín hiệu trong khung với hàm cửa sổ
        x_windowed = x_frame * win
        
        # Tính toán FFT của khung đã nhân cửa sổ (với zero-padding nếu n_fft > n_per_seg)
        fft_result = np.fft.fft(x_windowed, n=n_fft)
        
        # Chỉ lấy phần phổ tần số dương (One-sided) và chuẩn hóa biên độ hai phía
        # Chuẩn hóa phổ biên độ: nhân 2/N (trừ vạch DC và vạch Nyquist)
        Zxx[:, m] = fft_result[:n_freqs] * (2.0 / n_per_seg)
        Zxx[0, m] /= 2.0  # Vạch DC không nhân 2
        if n_fft % 2 == 0:
            Zxx[-1, m] /= 2.0  # Vạch Nyquist không nhân 2
            
    # 4. Tạo trục Tần số và Thời gian chính xác để hiển thị lên GUI
    f = np.arange(n_freqs) * (fs / n_fft)
    t = np.arange(n_frames) * (hop_size / fs) + (n_per_seg / (2.0 * fs))
    
    return f, t, Zxx

def get_spectrogram(Zxx):
    """
    Tính ma trận Spectrogram (mật độ phổ năng lượng) từ ma trận kết quả STFT phức.
    Trả về giá trị tính theo thang đo Decibel (dB) để hiển thị biểu đồ trực quan.
    """
    # Tính bình phương độ lớn
    magnitude = np.abs(Zxx)
    # Tránh lỗi log(0) bằng cách cộng thêm một lượng eps cực nhỏ
    spectrogram_db = 20 * np.log10(magnitude + 1e-10)
    return spectrogram_db

def istft_custom(Zxx, window_type='hamming', n_per_seg=512, hop_size=256, n_fft=512):
    """
    Biến đổi Fourier thời gian ngắn ngược (ISTFT) sử dụng thuật toán Chồng-Cộng (Overlap-Add).
    Hàm này dùng để kiểm chứng khả năng khôi phục tín hiệu miền thời gian từ ma trận phổ.
    """
    n_freqs, n_frames = Zxx.shape
    expected_len = (n_frames - 1) * hop_size + n_per_seg
    
    x_recon = np.zeros(expected_len)
    window_sum = np.zeros(expected_len)
    
    win = generate_window(window_type, n_per_seg)
    
    for m in range(n_frames):
        start_idx = m * hop_size
        end_idx = start_idx + n_per_seg
        
        # Khôi phục lại phổ đầy đủ (Two-sided FFT) từ phổ một phía
        fft_frame = np.zeros(n_fft, dtype=complex)
        
        # Đảo ngược bước chuẩn hóa ở hàm stft_custom
        spec_col = Zxx[:, m].copy()
        spec_col[0] *= 2.0
        if n_fft % 2 == 0:
            spec_col[-1] *= 2.0
        spec_col *= (n_per_seg / 2.0)
        
        fft_frame[:n_freqs] = spec_col
        if n_fft % 2 == 0:
            fft_frame[n_freqs:] = np.conj(spec_col[-2:0:-1])
        else:
            fft_frame[n_freqs:] = np.conj(spec_col[-1:0:-1])
            
        # Biến đổi IFFT để đưa về miền thời gian
        ifft_frame = np.real(np.fft.ifft(fft_frame))[:n_per_seg]
        
        # Phương pháp Chồng - Cộng (Overlap-Add)
        x_recon[start_idx:end_idx] += ifft_frame * win
        window_sum[start_idx:end_idx] += win ** 2
        
    # Chuẩn hóa lại bằng tổng bình phương cửa sổ để triệt tiêu hiệu ứng biên độ do đè khung
    # Tránh chia cho 0 ở những vùng biên ngoài rìa bằng cách set giá trị nhỏ về 1
    window_sum[window_sum < 1e-10] = 1.0
    x_recon /= window_sum
    
    return x_recon