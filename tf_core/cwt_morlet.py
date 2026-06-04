import numpy as np

def morlet_wavelet_ft(omega, scale, omega_0=5.0):
    """
    Tính toán biểu diễn trong miền tần số (Fourier Transform) của Wavelet Morlet phức
    đã được co giãn theo tham số 'scale'.
    
    Công thức giải tích miền tần số của Morlet Wavelet chuẩn hóa năng lượng:
    Psi_a(omega) = (4 * pi * scale^2)^(1/4) * exp(-0.5 * (scale * omega - omega_0)^2)
    """
    # Hệ số chuẩn hóa năng lượng bảo toàn qua các scale
    normalization = (4.0 * np.pi * (scale**2))**0.25
    
    # Biểu thức mũ hàm Gaussian miền tần số
    exponent = -0.5 * (scale * omega - omega_0)**2
    
    return normalization * np.exp(exponent)

def cwt_morlet(x, fs, nv=12, f_min=1.0, f_max=None, omega_0=5.0):
    """
    Triển khai Biến đổi Wavelet Liên tục (CWT) sử dụng Morlet Wavelet phức 
    thông qua phép dịch chập dựa trên FFT (FFT-based convolution).
    
    Tham số:
    - x: Mảng tín hiệu đầu vào thực hoặc phức (1D array)
    - fs: Tần số lấy mẫu của tín hiệu (Hz)
    - nv: Số lượng tỷ lệ trong một quãng tám (Voices per Octave) - điều khiển độ mịn trục Y
    - f_min: Tần số phân tích tối thiểu (Hz) (Mặc định: 1 Hz)
    - f_max: Tần số phân tích tối đa (Hz) (Mặc định: fs / 2 - tần số Nyquist)
    
    Trả về:
    - frequencies: Mảng chứa các giá trị tần số vật lý tương ứng với từng scale (Trục Y)
    - times: Mảng chứa các điểm thời gian (Trục X)
    - scalogram_complex: Ma trận CWT phức [Số lượng tần số x Số điểm thời gian]
    """
    if f_max is None:
        f_max = fs / 2.0
        
    n_samples = len(x)
    
    # 1. Tính toán dải tần số logarit và chuyển đổi sang dải Tỷ lệ (Scales)
    # CWT hoạt động tối ưu trên lưới tần số logarit (tương ứng với cấu trúc tai người)
    j_max = int(np.ceil(np.log2(f_max / f_min)))
    n_scales = j_max * nv
    
    # Tạo dải tần số phân bổ theo quy luật hình học (logarithmic spacing)
    frequencies = f_min * (2.0 ** (np.arange(n_scales) / nv))
    # Đảo ngược mảng tần số để tần số cao nằm ở trên cùng ma trận (tiện hiển thị đồ thị)
    frequencies = frequencies[::-1] 
    
    # Công thức ánh xạ: scale = (omega_0 * fs) / (2 * pi * f)
    scales = (omega_0 * fs) / (2.0 * np.pi * frequencies)
    
    # 2. Chuyển đổi tín hiệu đầu vào sang miền tần số bằng FFT
    # Thực hiện zero-padding lên lũy thừa của 2 kế tiếp để tối ưu tốc độ thuật toán FFT
    n_fft = int(2 ** np.ceil(np.log2(n_samples)))
    X_ft = np.fft.fft(x, n=n_fft)
    
    # Định nghĩa trục tần số góc (angular frequencies) tương ứng với mảng FFT
    # Trục này chạy từ 0 đến Nyquist rồi quay từ Nyquist âm về 0
    omega = np.fft.fftfreq(n_fft, d=1.0/fs) * 2.0 * np.pi
    
    # Khởi tạo ma trận rỗng chứa kết quả phức của Scalogram
    scalogram_complex = np.zeros((len(scales), n_samples), dtype=complex)
    
    # 3. Vòng lặp tính toán CWT cho từng Scale thông qua phép nhân miền tần số
    for idx, scale in enumerate(scales):
        # Lấy biểu diễn Fourier của Wavelet mẹ tại scale hiện tại
        Psi_ft = morlet_wavelet_ft(omega, scale, omega_0=omega_0)
        
        # Phép dịch chập trong miền thời gian <=> Phép nhân trực tiếp trong miền tần số
        # Do tích chập trong CWT sử dụng liên hợp phức của wavelet liên tục: x(t) * psi^*(-t/a)
        # Trong miền Fourier, điều này tương đương với nhân trực tiếp X(omega) * Psi^*(omega)
        Y_ft = X_ft * np.conj(Psi_ft)
        
        # Biến đổi ngược IFFT để đưa kết quả tích chập về miền thời gian
        y_time = np.fft.ifft(Y_ft)
        
        # Cắt bỏ phần zero-padding, chỉ lấy đúng chiều dài gốc của tín hiệu
        scalogram_complex[idx, :] = y_time[:n_samples]
        
    # 4. Thiết lập trục thời gian vật lý phục vụ đồ thị GUI
    times = np.arange(n_samples) / fs
    
    return frequencies, times, scalogram_complex

def get_scalogram_magnitude(scalogram_complex):
    """
    Tính toán mật độ năng lượng Wavelet (Scalogram Magnitude).
    Trả về giá trị ở dạng Decibel (dB) phục vụ cho hiển thị bản đồ màu trên GUI.
    """
    magnitude = np.abs(scalogram_complex)
    # Chuẩn hóa logarit tránh lỗi log(0)
    scalogram_db = 20 * np.log10(magnitude + 1e-10)
    return scalogram_db