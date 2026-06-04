'''
│
├── tf_core/                      # Lõi thuật toán biến đổi Thời gian - Tần số
│   ├── __init__.py
│   ├── stft_custom.py            # Tự lập trình STFT (chia khung, nhân cửa sổ, FFT)
│   ├── cwt_morlet.py             # Triển khai CWT qua phép nhân miền tần số (FFT-based convolve)
│   ├── wigner_ville.py           # Triển khai WVD thuần và Pseudo-WVD (bộ lọc Gaussian 2D)
│   └── synchrosqueezing.py       # Thuật toán tái phân bổ năng lượng Synchrosqueezed STFT
│
├── signal_generator/             # Trình tạo hoặc đọc tín hiệu thử nghiệm
│   ├── __init__.py
│   ├── synthetic.py              # Tạo xung Gaussian, linear chirp, FM chirp (tiếng dơi)
│   └── audio_loader.py           # Đọc file âm thanh WAV (tiếng nói, dữ liệu chuỗi số)
│
├── gui/                          # Giao diện đồ họa tương tác (Sinh viên A)
│   ├── __init__.py
│   ├── main_gui.py               # Cửa sổ chính chứa các Slider cấu hình tham số
│   └── canvas.py                 # Widget hiển thị bản đồ màu (Spectrogram, Scalogram) bằng Matplotlib
│
├── experiments/                  # Các kịch bản chạy mô phỏng để lấy số liệu báo cáo (Sinh viên B)
│   ├── heisenberg_test.py        # Thí nghiệm đo tích số thời gian-băng thông với xung Gaussian
│   ├── chirp_estimation.py       # So sánh độ chính xác ước lượng tần số tức thời
│   └── benchmark_perf.py         # Đo đạc thời gian tính toán và lập bảng so sánh
│
├── data/                         # Thư mục lưu trữ dữ liệu đầu vào và đầu ra
│   ├── input_signals/
│   │   ├── bat_call.wav          # Tiếng dơi kêu gốc
│   │   └── speech_5s.wav         # Đoạn tiếng nói 5 giây
│   └── saved_plots/              # Lưu các hình ảnh đồ thị xuất ra để chèn vào báo cáo
│
├── docs/                         # Báo cáo và tài liệu toán học
│   └── Joint_Report_TF_GroupX.pdf
│
├── app.py                        # Điểm khởi chạy chính của ứng dụng Workbench GUI
├── README.md                     # Hướng dẫn cài đặt thư viện và cách chạy chương trình
└── requirements.txt              # Danh sách thư viện: numpy, matplotlib, scipy, PyQt5 (hoặc PySide6)
'''

import os
from pathlib import Path

def create_workbench_structure():
    # Định nghĩa cấu trúc thư mục dưới dạng dictionary
    # Key là tên thư mục, value là danh sách các file nằm trong thư mục đó
    project_structure = {
        "tf_core": [
            "__init__.py",
            "stft_custom.py",
            "cwt_morlet.py",
            "wigner_ville.py",
            "synchrosqueezing.py"
        ],
        "signal_generator": [
            "__init__.py",
            "synthetic.py",
            "audio_loader.py"
        ],
        "gui": [
            "__init__.py",
            "main_gui.py",
            "canvas.py"
        ],
        "experiments": [
            "heisenberg_test.py",
            "chirp_estimation.py",
            "benchmark_perf.py"
        ],
        "data/input_signals": [],  # Thư mục rỗng chứa file âm thanh gốc
        "data/saved_plots": [],    # Thư mục rỗng chứa ảnh đồ thị xuất ra
        "docs": []                 # Thư mục rỗng chứa báo cáo
    }

    # Các file nằm ở thư mục gốc của dự án
    root_files = [
        "app.py",
        "README.md",
        "requirements.txt"
    ]

    print("🚀 Bắt đầu khởi tạo cấu trúc thư mục bài tập lớn...")

    # 1. Tạo các thư mục và file con bên trong
    for folder, files in project_structure.items():
        folder_path = Path(folder)
        # Tạo thư mục (bao gồm cả thư mục cha nếu chưa có như data/input_signals)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Đã tạo thư mục: {folder_path}")

        # Tạo các file rỗng bên trong thư mục
        for file in files:
            file_path = folder_path / file
            file_path.touch(exist_ok=True)
            print(f"  📄 Đã tạo file: {file_path}")

    # 2. Tạo các file ở thư mục gốc
    for file in root_files:
        file_path = Path(file)
        file_path.touch(exist_ok=True)
        print(f"📄 Đã tạo file gốc: {file_path}")

    # 3. Ghi sẵn một số thư viện cần thiết vào file requirements.txt
    requirements_content = "numpy\nscipy\nmatplotlib\nPyQt5\n"
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print("📝 Đã tự động điền các thư viện cơ bản vào requirements.txt")

    print("\n✅ Hoàn thành! Cấu trúc thư mục 'Time-Frequency Analysis Workbench' đã sẵn sàng để code.")

if __name__ == "__main__":
    create_workbench_structure()