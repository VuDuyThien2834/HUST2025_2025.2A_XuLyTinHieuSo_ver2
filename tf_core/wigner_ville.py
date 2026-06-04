import numpy as np
from scipy.signal import hilbert

def wigner_ville_pure(x, fs):
    """
    Triển khai Phân bố Wigner-Ville thuần túy (Pure WVD) cho tín hiệu.
    Bắt buộc áp dụng biến đổi Hilbert để chuyển tín hiệu thực thành tín hiệu giải tích.
    
    Tham số:
    - x: Mảng tín hiệu đầu vào (1D array)
    - fs: Tần số lấy mẫu (Hz)
    
    Trả về:
    - f: Mảng chứa các vạch tần số từ 0 đến fs/2 (Trục Y)
    - t: Mảng chứa các điểm thời gian (Trục X)
    - wvd_matrix: Ma trận WVD thực [Số vạch tần số x Số điểm thời gian]
    """
    n_samples = len(x)
    
    # Bước 1: Bắt buộc chuyển đổi sang Tín hiệu giải tích để triệt tiêu tần số âm
    # Việc này giúp loại bỏ nhiễu chéo sinh ra tại vạch 0 Hz giữa tần số âm và dương
    z = hilbert(x)
    
    # Khởi tạo ma trận chứa Hàm tự tương quan tức thời (Instantaneous Autocorrelation Function - IAF)
    # Kích thước ma trận: [N x N] trong đó hàng đại diện cho độ trễ tau, cột đại diện cho thời gian t
    iaf = np.zeros((n_samples, n_samples), dtype=complex)
    
    # Bước 2: Vòng lặp tính toán IAF toán tử bậc hai
    # R_z(t, tau) = z(t + tau/2) * conj(z(t - tau/2))
    for t_idx in range(n_samples):
        # Xác định khoảng trễ tau tối đa khả thi tại điểm thời gian t_idx để không vượt quá biên mảng
        max_tau = min(t_idx, n_samples - 1 - t_idx)
        
        # Chỉ tính toán các giá trị trễ nằm trong vùng an toàn
        tau_range = np.arange(-max_tau, max_tau + 1)
        
        # Ánh xạ chỉ số để tính toán đối xứng xung quanh tâm t_idx
        idx_plus = t_idx + tau_range
        idx_minus = t_idx - tau_range
        
        # Tính toán giá trị tự tương quan tức thời
        # Lưu vào ma trận IAF (phần trễ tau âm được xử lý dịch tuần hoàn hoặc xếp mảng)
        iaf[tau_range % n_samples, t_idx] = z[idx_plus] * np.conj(z[idx_minus])
        
    # Bước 3: Biến đổi Fourier (FFT) theo trục độ trễ tau (trục dọc - axis 0)
    # Kết quả của WVD theo định nghĩa lý thuyết luôn luôn là một ma trận số thực thuần túy
    wvd_matrix = np.real(np.fft.fft(iaf, axis=0))
    
    # Do thuật toán lấy bước nhảy đối xứng tau/2, dải tần số của WVD gốc bị giãn gấp đôi (0 đến fs)
    # Chúng ta chỉ lấy nửa đầu ma trận ứng với dải tần vật lý thực tế từ 0 đến fs/2 (Nyquist)
    n_freqs = n_samples // 2 + 1
    wvd_matrix = wvd_matrix[:n_freqs, :]
    
    # Thiết lập các trục tọa độ vật lý cho đồ thị
    f = np.linspace(0, fs / 2.0, n_freqs)
    t = np.arange(n_samples) / fs
    
    return f, t, wvd_matrix

def gaussian_kernel_2d(sigma_t, sigma_f, num_t, num_f):
    """
    Tạo một cửa sổ nhân làm mịn Gaussian 2D (Smoothing Kernel) 
    phục vụ cho việc triệt tiêu nhiễu chéo trong cấu trúc Pseudo-WVD.
    """
    t = np.arange(-num_t // 2, num_t // 2 + 1)
    f = np.arange(-num_f // 2, num_f // 2 + 1)
    t_grid, f_grid = np.meshgrid(t, f)
    
    # Công thức phân bố Gaussian hai chiều chuẩn hóa
    kernel = np.exp(-0.5 * ((t_grid / sigma_t)**2 + (f_grid / sigma_f)**2))
    return kernel / np.sum(kernel)

def pseudo_wigner_ville(x, fs, sigma_t=4.0, sigma_f=4.0):
    """
    Triển khai Phân bố Pseudo-Wigner-Ville (PWVD).
    Sử dụng kỹ thuật nhân chập ma trận hai chiều với nhân Gaussian 
    để dập tắt các dao động tần số cao của thành phần nhiễu chéo (cross-terms).
    
    Tham số:
    - x: Tín hiệu đầu vào 1D
    - fs: Tần số lấy mẫu (Hz)
    - sigma_t: Độ rộng bộ lọc làm mịn theo trục Thời gian (giá trị càng lớn lọc càng mạnh)
    - sigma_f: Độ rộng bộ lọc làm mịn theo trục Tần số
    """
    # 1. Tính toán ma trận WVD thuần túy làm nền tảng
    f, t, wvd_pure = wigner_ville_pure(x, fs)
    
    # 2. Định nghĩa kích thước cửa sổ lọc nhân Gaussian 2D dựa trên quy tắc 3-sigma
    num_t = int(6 * sigma_t) | 1  # Đảm bảo kích thước cửa sổ là số lẻ
    num_f = int(6 * sigma_f) | 1
    
    kernel = gaussian_kernel_2d(sigma_t, sigma_f, num_t, num_f)
    
    # 3. Thực hiện phép nhân chập ma trận 2D sử dụng thư viện xử lý mảng nâng cao
    # Để giữ nguyên kích thước ma trận gốc, ta áp dụng chế độ padding 'same'
    from scipy.signal import convolve2d
    wvd_smoothed = convolve2d(wvd_pure, kernel, mode='same', boundary='symm')
    
    return f, t, wvd_smoothed

def get_wvd_db(wvd_matrix):
    """
    Chuẩn hóa ma trận WVD sang thang đo Decibel (dB) để hiển thị bản đồ năng lượng.
    Do WVD có thể sinh ra các giá trị âm nhỏ nhiễu toán học, chúng ta cần loại bỏ phần âm trước khi tính log.
    """
    wvd_positive = np.maximum(wvd_matrix, 0.0)
    wvd_db = 10 * np.log10(wvd_positive + 1e-10)
    return wvd_db