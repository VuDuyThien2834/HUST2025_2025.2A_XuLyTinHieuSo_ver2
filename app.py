import sys
from PyQt5.QtWidgets import QApplication
from gui.main_gui import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    workbench = MainWindow()
    workbench.show()
    sys.exit(app.exec_())