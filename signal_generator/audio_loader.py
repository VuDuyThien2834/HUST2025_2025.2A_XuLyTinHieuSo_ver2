import numpy as np
from scipy.io import wavfile

def load_wav_signal(file_path, max_duration=5.0):
    """
    Tải file âm thanh định dạng `.wav`, chuẩn hóa biên độ và đưa về kênh Mono.
    Hỗ trợ cắt ngắn tín hiệu theo thời gian tối đa để đảm bảo hiệu năng tính toán.
    
    Tham số:
    - file_path: Đường dẫn tới file .wav cần đọc (String hoặc Path object)
    - max_duration: Thời gian tối đa muốn lấy (giây) (Mặc định: 5.0 giây theo yêu cầu đề tài)
    
    Trả về:
    - fs: Tần số lấy mẫu của file âm thanh (Hz)
    - t: Mảng thời gian (1D array)
    - x: Mảng dữ liệu tín hiệu đã chuẩn hóa về dải [-1.0, 1.0] (1D array)
    """
    try:
        # 1. Đọc dữ liệu từ file WAV bằng thư viện scipy
        fs, data = wavfile.read(file_path)
        
        # 2. Xử lý trường hợp âm thanh nhiều kênh (Stereo / Multi-channel)
        # Nếu ma trận dữ liệu có 2 chiều, tiến hành lấy trung bình các kênh để đưa về Mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            
        # 3. Chuẩn hóa biên độ (Amplitude Normalization)
        # File WAV có thể lưu ở dạng số nguyên 16-bit (int16), 32-bit (int32) hoặc số thực (float32)
        # Ta cần ép kiểu về float và đưa biên độ về khoảng [-1.0, 1.0] để các thuật toán DSP không bị tràn số
        if data.dtype == np.int16:
            x = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            x = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            # Dữ liệu 8-bit không dấu có dải từ 0 đến 255, tâm nằm ở 128
            x = (data.astype(np.float32) - 128.0) / 128.0
        else:
            # Nếu đã là dạng float, giữ nguyên hoặc chuẩn hóa theo đỉnh năng lượng tối đa
            max_val = np.max(np.abs(data))
            x = data.astype(np.float32) / (max_val if max_val > 0 else 1.0)
            
        # 4. Giới hạn thời gian tín hiệu (Cắt đúng đoạn 5 giây đầu hoặc theo max_duration)
        max_samples = int(fs * max_duration)
        if len(x) > max_samples:
            x = x[:max_samples]
            
        # 5. Tạo trục thời gian tương ứng cho tín hiệu
        n_samples = len(x)
        t = np.arange(n_samples) / fs
        
        print(f"🎵 Đã tải thành công file: {file_path}")
        print(f"   - Tần số lấy mẫu (Fs): {fs} Hz")
        print(f"   - Tổng số mẫu: {n_samples}")
        print(f"   - Thời lượng thực tế: {n_samples / fs:.2f} giây")
        
        return fs, t, x

    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Không tìm thấy file âm thanh tại đường dẫn: {file_path}")
    except Exception as e:
        raise Exception(f"❌ Có lỗi xảy ra khi đọc file WAV: {str(e)}")

def save_wav_signal(file_path, fs, x):
    """
    Hàm bổ trợ: Lưu mảng tín hiệu số 1D ngược trở lại thành file `.wav` 16-bit.
    Hàm này hữu ích nếu nhóm muốn xuất file âm thanh sau khi xử lý hoặc lọc nhiễu.
    """
    # Đưa tín hiệu từ dải [-1.0, 1.0] về định dạng số nguyên int16 chuẩn
    # Giới hạn giá trị trong khoảng an toàn để tránh hiện tượng méo tiếng (clipping)
    x_clipped = np.clip(x, -1.0, 1.0)
    data_int16 = (x_clipped * 32767.0).astype(np.int16)
    
    # Ghi file
    wavfile.write(file_path, fs, data_int16)
    print(f"💾 Đã lưu tín hiệu vào file: {file_path}")