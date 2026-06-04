import numpy as np
import matplotlib.pyplot as plt
from signal_generator.synthetic import generate_gaussian_pulse

def calculate_uncertainty_product(fs, sigma):
    """
    Tính toán định lượng delta_t, delta_f và tích số bất định của xung Gaussian.
    """
    duration = 2.0  # Tăng thời gian để xung Gaussian suy hao hoàn toàn về 0 ở biên
    t, x = generate_gaussian_pulse(fs, sigma=sigma, duration=duration)
    
    # Chuẩn hóa năng lượng tín hiệu miền thời gian về 1
    dt = 1.0 / fs
    energy_t = np.sum(np.abs(x)**2) * dt
    x = x / np.sqrt(energy_t)
    
    # 1. Tính toán miền Thời gian: t_0 và delta_t
    t_0 = np.sum(t * (np.abs(x)**2)) * dt
    delta_t = np.sqrt(np.sum(((t - t_0)**2) * (np.abs(x)**2)) * dt)
    
    # 2. Tính toán miền Tần số thông qua FFT
    n_fft = len(x)
    X_f = np.fft.fft(x) * dt
    f = np.fft.fftfreq(n_fft, d=dt)
    
    # Sắp xếp lại trục tần số từ âm đến dương để tính tích phân chuẩn xác
    idx_sort = np.argsort(f)
    f = f[idx_sort]
    X_f = X_f[idx_sort]
    
    # Chuẩn hóa năng lượng miền tần số
    df = f[1] - f[0]
    energy_f = np.sum(np.abs(X_f)**2) * df
    
    # Tính f_0 và delta_f
    f_0 = np.sum(f * (np.abs(X_f)**2)) * df
    delta_f = np.sqrt(np.sum(((f - f_0)**2) * (np.abs(X_f)**2)) * df)
    
    product = delta_t * delta_f
    return delta_t, delta_f, product

def run_heisenberg_experiment():
    fs = 1000
    sigmas = np.linspace(0.02, 0.15, 10)
    
    results_dt = []
    results_df = []
    results_prod = []
    
    print("📊 Đang chạy thực nghiệm kiểm chứng hệ thức Heisenberg-Gabor...")
    print(f"{'Sigma (s)':<12}{'Delta t (s)':<12}{'Delta f (Hz)':<12}{'Tích số (dt*df)':<15}{'Giới hạn (1/4pi)':<15}")
    print("-" * 66)
    
    for sigma in sigmas:
        dt, df, prod = calculate_uncertainty_product(fs, sigma)
        results_dt.append(dt)
        results_df.append(df)
        results_prod.append(prod)
        print(f"{sigma:<12.3f}{dt:<12.4f}{df:<12.4f}{prod:<15.5f}{1/(4*np.pi):<15.5f}")
        
    # Vẽ đồ thị kết quả thực nghiệm để chèn vào mục 1.6 của báo cáo
    plt.figure(figsize=(10, 5))
    plt.plot(sigmas, results_prod, 'o-', label='Tích số thực nghiệm ($\Delta t \cdot \Delta f$)', color='red')
    plt.axhline(y=1/(4*np.pi), color='blue', linestyle='--', label='Giới hạn lý thuyết Gabor ($1/4\pi \approx 0.0796$)')
    plt.title("Thực nghiệm kiểm chứng Nguyên lý bất định Heisenberg-Gabor với Xung Gaussian")
    plt.xlabel("Tham số độ rộng xung Sigma ($\sigma$)")
    plt.ylabel("Giá trị Tích số Độ phân giải")
    plt.grid(True, alpha=0.4)
    plt.legend()
    
    # Tự động tạo thư mục chứa ảnh nếu chưa có và lưu đồ thị
    os.makedirs("data/saved_plots", exist_ok=True)
    plt.savefig("data/saved_plots/heisenberg_verification.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_heisenberg_experiment()