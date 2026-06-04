import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QSlider, 
                             QComboBox, QPushButton, QFileDialog, QGroupBox)
from PyQt5.QtCore import Qt

# Import các hàm sinh tín hiệu và các thuật toán từ các module đã lập trình
from signal_generator.synthetic import generate_linear_chirp, generate_gaussian_pulse, generate_bat_call
from signal_generator.audio_loader import load_wav_signal
from tf_core.stft_custom import stft_custom, get_spectrogram
from tf_core.cwt_morlet import cwt_morlet, get_scalogram_magnitude
from tf_core.wigner_ville import wigner_ville_pure, pseudo_wigner_ville, get_wvd_db
from tf_core.synchrosqueezing import synchrosqueezed_stft, get_sst_db
from gui.canvas import TFCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time-Frequency Analysis Workbench (STFT, CWT, WVD, SST)")
        self.resize(1280, 800)
        
        # Thiết lập các tham số tín hiệu mặc định
        self.fs = 8000
        self.t = np.array([])
        self.x = np.array([])
        self.current_signal_type = "Linear Chirp"
        self.loaded_file_path = ""
        
        # Khởi tạo giao diện chính
        self.init_ui()
        # Sinh tín hiệu mặc định ban đầu và chạy tính toán phổ
        self.update_signal()

    def init_ui(self):
        # 1. Khung chứa chính (Main Widget)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 2. Panel bên trái: Các thanh điều khiển và cấu hình (Kích thước cố định)
        control_panel = QWidget()
        control_panel.setFixedWidth(320)
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)
        
        # --- Nhóm 1: Chọn Tín hiệu Đầu vào ---
        signal_group = QGroupBox("1. Tín hiệu đầu vào")
        signal_grid = QGridLayout(signal_group)
        
        signal_grid.addWidget(QLabel("Loại tín hiệu:"), 0, 0)
        self.combo_signal = QComboBox()
        self.combo_signal.addItems(["Linear Chirp", "Gaussian Pulse", "Bat Call (FM Chirp)", "Tải file .WAV"])
        self.combo_signal.currentIndexChanged.connect(self.on_signal_type_changed)
        signal_grid.addWidget(self.combo_signal, 0, 1)
        
        self.btn_browse = QPushButton("Chọn file WAV...")
        self.btn_browse.setEnabled(False)
        self.btn_browse.clicked.connect(self.browse_wav)
        signal_grid.addWidget(self.btn_browse, 1, 0, 1, 2)
        
        control_layout.addWidget(signal_group)
        
        # --- Nhóm 2: Tham số Thuật toán Phân tích ---
        method_group = QGroupBox("2. Phương pháp phân tích")
        method_vbox = QVBoxLayout(method_group)
        
        method_vbox.addWidget(QLabel("Chọn phương pháp hiển thị:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["STFT (Spectrogram)", "CWT (Scalogram)", "Wigner-Ville (Pure WVD)", "Pseudo-WVD (Làm mịn)", "Synchrosqueezed STFT (SST)"])
        # self.combo_method.currentIndexChanged.connect(self.recompute_and_plot)
        self.combo_method.currentIndexChanged.connect(self.on_param_changed)
        method_vbox.addWidget(self.combo_method)
        
        control_layout.addWidget(method_group)
        
        # --- Nhóm 3: Các Slider tinh chỉnh tham số ---
        param_group = QGroupBox("3. Tinh chỉnh tham số")
        param_grid = QGridLayout(param_group)
        
        # Slider 1: Chiều dài cửa sổ (Dùng cho STFT, SST)
        param_grid.addWidget(QLabel("Chiều dài cửa sổ STFT (N):"), 0, 0)
        self.lbl_win_len = QLabel("512")
        param_grid.addWidget(self.lbl_win_len, 0, 1, Qt.AlignRight)
        self.slider_win_len = QSlider(Qt.Horizontal)
        self.slider_win_len.setRange(6, 11) # 2^6=64 đến 2^11=2048
        self.slider_win_len.setValue(9)     # Mặc định 2^9 = 512
        self.slider_win_len.valueChanged.connect(self.on_param_changed)
        param_grid.addWidget(self.slider_win_len, 1, 0, 1, 2)
        
        # Slider 2: Tham số hóa hạt nhân Gaussian / Wavelet Voices (Dùng cho CWT hoặc PWVD)
        self.lbl_param_dynamic = QLabel("Tham số bổ trợ (CWT/PWVD):")
        param_grid.addWidget(self.lbl_param_dynamic, 2, 0)
        self.lbl_val_dynamic = QLabel("12")
        param_grid.addWidget(self.lbl_val_dynamic, 2, 1, Qt.AlignRight)
        self.slider_dynamic = QSlider(Qt.Horizontal)
        self.slider_dynamic.setRange(4, 32)
        self.slider_dynamic.setValue(12) # Mặc định 12 Voices hoặc sigma = 1.2
        self.slider_dynamic.valueChanged.connect(self.on_param_changed)
        param_grid.addWidget(self.slider_dynamic, 3, 0, 1, 2)
        
        control_layout.addWidget(param_group)
        control_layout.addStretch() # Đẩy các nhóm lên trên cùng
        
        # 3. Panel bên phải: Khu vực Đồ thị hiển thị (Matplotlib Canvas)
        self.canvas = TFCanvas()
        main_layout.addWidget(self.canvas, stretch=1)

    def on_signal_type_changed(self, index):
        self.current_signal_type = self.combo_signal.currentText()
        if self.current_signal_type == "Tải file .WAV":
            self.btn_browse.setEnabled(True)
        else:
            self.btn_browse.setEnabled(False)
            self.update_signal()

    def browse_wav(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file âm thanh WAV", "", "Wave Files (*.wav)")
        if file_path:
            self.loaded_file_path = file_path
            self.update_signal()

    def update_signal(self):
        """Khởi tạo hoặc cập nhật mảng tín hiệu x(t) khi người dùng đổi tùy chọn"""
        if self.current_signal_type == "Linear Chirp":
            self.fs = 8000
            self.t, self.x, _ = generate_linear_chirp(self.fs, duration=1.0, f0=100, f1=3500)
        elif self.current_signal_type == "Gaussian Pulse":
            self.fs = 8000
            self.t, self.x = generate_gaussian_pulse(self.fs, sigma=0.04, duration=1.0)
        elif self.current_signal_type == "Bat Call (FM Chirp)":
            self.fs = 16000
            self.t, self.x, _ = generate_bat_call(self.fs, duration=0.1)
        elif self.current_signal_type == "Tải file .WAV":
            if os.path.exists(self.loaded_file_path):
                self.fs, self.t, self.x = load_wav_signal(self.loaded_file_path, max_duration=5.0)
            else:
                return # Chưa chọn file thì chưa làm gì cả
                
        self.recompute_and_plot()

    def on_param_changed(self):
        # 1. Cập nhật số hiển thị của Chiều dài cửa sổ
        actual_win_len = 2 ** self.slider_win_len.value()
        self.lbl_win_len.setText(str(actual_win_len))
        
        # 2. Đọc phương pháp hiện tại đang chọn trên giao diện
        method = self.combo_method.currentText()
        
        # 3. Thay đổi động tên và trạng thái của các trường tinh chỉnh
        if "CWT" in method:
            self.lbl_param_dynamic.setText("Số vạch bộ lọc CWT (Voices):")
            self.lbl_val_dynamic.setText(str(self.slider_dynamic.value()))
            self.slider_dynamic.setEnabled(True) # Mở khóa thanh trượt số 2
        elif "Pseudo-WVD" in method:
            self.lbl_param_dynamic.setText("Hệ số làm mịn (Sigma 2D x10):")
            self.lbl_val_dynamic.setText(f"{self.slider_dynamic.value() / 10.0:.1f}")
            self.slider_dynamic.setEnabled(True) # Mở khóa thanh trượt số 2
        else:
            # Đối với STFT hoặc SST hoặc WVD thuần
            self.lbl_param_dynamic.setText("Hệ số bổ trợ (Không dùng):")
            self.lbl_val_dynamic.setText("-")
            self.slider_dynamic.setEnabled(False) # Khóa thanh trượt số 2 lại
            
        # 4. Sau khi cập nhật giao diện xong mới tiến hành tính toán và vẽ lại đồ thị
        self.recompute_and_plot()

    def recompute_and_plot(self):
        """Khối lõi thực hiện liên kết giao diện và các file tính toán toán học"""
        if len(self.x) == 0:
            return
            
        method = self.combo_method.currentText()
        n_per_seg = 2 ** self.slider_win_len.value()
        hop_size = n_per_seg // 2 # Mặc định overlap 50%
        
        # Xóa đồ thị cũ để vẽ lại đồ thị mới
        self.canvas.clear_plots()
        
        # Luôn vẽ đồ thị miền thời gian của tín hiệu gốc lên trục trên (ax_time)
        self.canvas.plot_time_domain(self.t, self.x)
        
        # Nhánh rẽ tính toán ma trận tương ứng với lựa chọn của ComboBox
        if method == "STFT (Spectrogram)":
            f, t_spec, Zxx = stft_custom(self.x, self.fs, window_type='hamming', n_per_seg=n_per_seg, hop_size=hop_size, n_fft=n_per_seg)
            tf_data = get_spectrogram(Zxx)
            self.canvas.plot_time_frequency(t_spec, f, tf_data, "Spectrogram (STFT) - Thang đo dB", cmap='jet')
            
        elif method == "CWT (Scalogram)":
            voices = self.slider_dynamic.value()
            f, t_cwt, scalogram_complex = cwt_morlet(self.x, self.fs, nv=voices, f_min=10.0, f_max=self.fs/2.0)
            tf_data = get_scalogram_magnitude(scalogram_complex)
            self.canvas.plot_time_frequency(t_cwt, f, tf_data, "Scalogram (CWT Morlet) - Thang đo dB", cmap='viridis', y_scale_log=True)
            
        elif method == "Wigner-Ville (Pure WVD)":
            # WVD tính toán trên toàn bộ số điểm mẫu, giới hạn độ dài tín hiệu để tránh treo máy đồ thị
            x_truncated = self.x[:1000] if len(self.x) > 1000 else self.x
            f, t_wvd, wvd_matrix = wigner_ville_pure(x_truncated, self.fs)
            tf_data = get_wvd_db(wvd_matrix)
            self.canvas.plot_time_frequency(t_wvd, f, tf_data, "Pure Wigner-Ville Distribution (WVD) - dB", cmap='inferno')
            
        elif method == "Pseudo-WVD (Làm mịn)":
            x_truncated = self.x[:1000] if len(self.x) > 1000 else self.x
            sigma_val = self.slider_dynamic.value() / 10.0
            f, t_wvd, wvd_matrix = pseudo_wicorn_ville = pseudo_wigner_ville(x_truncated, self.fs, sigma_t=sigma_val, sigma_f=sigma_val)
            tf_data = get_wvd_db(wvd_matrix)
            self.canvas.plot_time_frequency(t_wvd, f, tf_data, f"Pseudo-WVD (Gaussian Smoothed, Sigma={sigma_val}) - dB", cmap='inferno')
            
        elif method == "Synchrosqueezed STFT (SST)":
            f, t_sst, sst_matrix = synchrosqueezed_stft(self.x, self.fs, window_type='hamming', n_per_seg=n_per_seg, hop_size=hop_size, n_fft=n_per_seg)
            tf_data = get_sst_db(sst_matrix)
            self.canvas.plot_time_frequency(t_sst, f, tf_data, "Synchrosqueezed STFT (SST) - Thang đo dB", cmap='hot')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_copy() if hasattr(app, 'exec_copy') else app.exec_())