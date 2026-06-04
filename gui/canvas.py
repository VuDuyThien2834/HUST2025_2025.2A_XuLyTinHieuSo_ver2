import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # Ép cấu hình sử dụng backend kết nối PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TFCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        # Khởi tạo Figure chứa hai đồ thị xếp chồng theo chiều dọc
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax_time = self.fig.add_subplot(211) # Đồ thị 1: Miền thời gian
        self.ax_tf = self.fig.add_subplot(212)   # Đồ thị 2: Miền Thời gian - Tần số
        self.fig.tight_layout(pad=3.5)
        
        super().__init__(self.fig)
        self.setParent(parent)
        self.colorbar = None
        self.cax = None # Quản lý trục riêng của colorbar để tránh crash khi remove

    def clear_plots(self):
        """Xóa sạch nội dung cũ trên các trục đồ thị trước khi vẽ mới"""
        self.ax_time.clear()
        self.ax_tf.clear()
        
        # Xóa trục chứa thanh màu cũ một cách an toàn
        if self.cax:
            self.cax.remove()
            self.cax = None
        self.colorbar = None

    def plot_time_domain(self, t, x):
        """Vẽ biểu đồ dạng sóng thô của tín hiệu theo thời gian (Hàm bị thiếu của bạn)"""
        self.ax_time.plot(t, x, color='b', linewidth=1)
        self.ax_time.set_title("Dạng sóng tín hiệu miền Thời gian (Time Domain)")
        self.ax_time.set_xlabel("Thời gian (giây)")
        self.ax_time.set_ylabel("Biên độ (V)")
        self.ax_time.grid(True, alpha=0.3)
        if len(t) > 0:
            self.ax_time.set_xlim(t[0], t[-1])

    def plot_time_frequency(self, t, f, tf_matrix, title, cmap='jet', y_scale_log=False):
        """Vẽ bản đồ màu mật độ năng lượng 2D lên mặt phẳng Thời gian - Tần số"""
        mesh = self.ax_tf.pcolormesh(t, f, tf_matrix, shading='gouraud', cmap=cmap)
        
        self.ax_tf.set_title(title)
        self.ax_tf.set_xlabel("Thời gian (giây)")
        self.ax_tf.set_ylabel("Tần số (Hz)")
        
        if y_scale_log:
            self.ax_tf.set_yscale('log')
            self.ax_tf.set_ylim(f[-1], f[0]) 
        else:
            self.ax_tf.set_ylim(0, np.max(f))
            
        if len(t) > 0:
            self.ax_tf.set_xlim(t[0], t[-1])
        
        # Tạo một trục cax độc lập nằm bên phải trục đồ thị chính để chứa thanh màu colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.ax_tf)
        self.cax = divider.append_axes("right", size="5%", pad=0.1)
        
        self.colorbar = self.fig.colorbar(mesh, cax=self.cax, orientation='vertical')
        self.colorbar.set_label('Cường độ Năng lượng (dB)')
        
        self.draw()