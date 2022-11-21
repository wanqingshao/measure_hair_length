from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QLineEdit, QPushButton, QFormLayout, QCheckBox
from PyQt5.QtGui import QIntValidator
from os.path import expanduser
from PyQt5.QtCore import Qt

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.input_path = ""
        self.output_path = ""
        self.minlength_val = 200
        self.keep_edges = False
        self.iniUI()
        

    def iniUI(self):
        outerLayout = QVBoxLayout()
        directory_layout = QVBoxLayout()
        min_length_layout = QFormLayout()
        edge_layout = QVBoxLayout()
        run_layout = QVBoxLayout()

        self.button1 = QPushButton(self)
        self.button1.setFixedSize(150, 30)
        self.button2 = QPushButton(self)
        self.button2.setFixedSize(150, 30)
        self.label1 = QLabel(self) 
        self.label2 = QLabel(self) 
        self.button1.setText("select input folder")
        self.button1.clicked.connect(self.pick_input_folder)  # add (connect) button event 
        self.button2.setText("select output folder")
        self.button2.clicked.connect(self.pick_output_folder)
        self.minlength = QLineEdit("200")
        self.minlength.setValidator(QIntValidator())
        self.minlength.setAlignment(Qt.AlignLeft)
        self.minlength.textChanged.connect(self.collect_min_length)
        self.edge = QCheckBox("Consider hair touch image edge")
        self.edge.toggled.connect(self.collect_edge_state)
        self.button3 = QPushButton(self)
        self.button3.setFixedSize(100, 30)
        self.button3.setText("Run")
        self.button3.clicked.connect(self.measure_length)

        directory_layout.addWidget(self.button1)
        directory_layout.addWidget(self.label1)
        directory_layout.addWidget(self.button2)
        directory_layout.addWidget(self.label2)
        min_length_layout.addRow("Minimum length to consider (pixels):", self.minlength)
        min_length_layout.setFormAlignment(Qt.AlignLeft)
        edge_layout.addWidget(self.edge)
        run_layout.addWidget(self.button3)

        outerLayout.addLayout(directory_layout)
        outerLayout.addLayout(min_length_layout)
        outerLayout.addLayout(edge_layout)
        outerLayout.addLayout(run_layout)
        outerLayout.addStretch()
        self.setLayout(outerLayout)

    
    def pick_input_folder(self):
        dialog = QFileDialog()
        folder_path = str(dialog.getExistingDirectory(self, "Open a folder",expanduser("~"), QFileDialog.ShowDirsOnly))
        self.input_path = folder_path
        self.label1.setText(folder_path)
        self.label1.show()
    
    def pick_output_folder(self):
        dialog = QFileDialog()
        folder_path = str(dialog.getExistingDirectory(self, "Open a folder",expanduser("~"), QFileDialog.ShowDirsOnly))
        self.output_path = folder_path
        self.label2.setText(folder_path)
        self.label2.show()

    def collect_min_length(self):
        self.minlength_val = int(self.minlength.text())


    def collect_edge_state(self):
        self.keep_edges = self.edge.isChecked()

    def measure_length(self):
        print(self.input_path)
        print(self.output_path)
        print(self.minlength_val)
        print(self.keep_edges)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.resize(400,200)
    window.setWindowTitle("MaoMao")
    window.show()
    app.exec_()