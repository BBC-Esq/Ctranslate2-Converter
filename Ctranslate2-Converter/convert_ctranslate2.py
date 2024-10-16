import sys
import os
from pathlib import Path
import logging
import traceback

from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QCheckBox, QTextEdit, QStyleFactory
from PySide6.QtCore import QThread, Signal
import subprocess

def set_cuda_paths():
    try:
        venv_base = Path(sys.executable).parent
        nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
        for env_var in ['CUDA_PATH', 'CUDA_PATH_V12_1', 'PATH']:
            current_path = os.environ.get(env_var, '')
            os.environ[env_var] = os.pathsep.join(filter(None, [str(nvidia_base_path), current_path]))
        logging.info("CUDA paths set successfully")
    except Exception as e:
        logging.error(f"Error setting CUDA paths: {str(e)}")
        logging.debug(traceback.format_exc())

set_cuda_paths()

class ConversionThread(QThread):
    started = Signal(str, str)
    finished = Signal(str, str)
    error = Signal(str)

    def __init__(self, command, quantization):
        super().__init__()
        self.command = command
        self.quantization = quantization

    def run(self):
        self.started.emit(self.quantization, self.command)
        current_env = os.environ.copy()
        result = subprocess.run(self.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=current_env)
        if result.returncode == 0:
            self.finished.emit(self.quantization, result.stdout)
        else:
            self.error.emit(f"Command failed with return code {result.returncode}: {result.stderr}")

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("chintellalaw.com - for non-commercial use")
        self.model_path = ""
        self.output_dir = ""
        self.resize(800, 500)
        layout = QVBoxLayout(self)

        browse_layout = QHBoxLayout()
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse)
        browse_layout.addWidget(self.browse_btn)

        self.path_label = QLabel("")
        browse_layout.addWidget(self.path_label)
        layout.addLayout(browse_layout)

        output_browse_layout = QHBoxLayout()
        self.output_browse_btn = QPushButton("Select Output Directory")
        self.output_browse_btn.clicked.connect(self.browse_output)
        output_browse_layout.addWidget(self.output_browse_btn)

        self.output_path_label = QLabel("")
        output_browse_layout.addWidget(self.output_path_label)
        layout.addLayout(output_browse_layout)

        self.quantization_options = ["float32", "float16", "bfloat16", "int8_float32", "int8_float16", "int8_bfloat16", "int8"]
        self.quant_vars = {option: QCheckBox(option) for option in self.quantization_options}

        quant_layout = QHBoxLayout()
        for option, chk in self.quant_vars.items():
            chk.setChecked(False)
            quant_layout.addWidget(chk)
        
        # Add AWQ checkbox
        self.awq_checkbox = QCheckBox("AWQ")
        quant_layout.addWidget(self.awq_checkbox)
        
        layout.addLayout(quant_layout)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_conversion)
        layout.addWidget(self.run_btn)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.conversion_queue = []
        self.current_conversion = None

    def browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if path:
            self.model_path = path
            self.path_label.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir = path
            self.output_path_label.setText(f"Output Directory: {path}")

    def append_to_text_widget(self, content):
        self.output_text.append(content)

    def run_conversion(self):
        self.output_text.clear()
        self.conversion_queue.clear()
        if not self.model_path:
            QMessageBox.critical(self, "Error", "Please select a model directory.")
            return
        if not self.output_dir:
            QMessageBox.critical(self, "Error", "Please select an output directory.")
            return

        if self.awq_checkbox.isChecked():
            self.conversion_queue.append("awq")
        
        for option, chk in self.quant_vars.items():
            if chk.isChecked():
                self.conversion_queue.append(option)
        self.process_next_conversion()

    def process_next_conversion(self):
        if self.conversion_queue:
            option = self.conversion_queue.pop(0)
            model_dir = self.model_path
            model_name = os.path.basename(model_dir)
            output_dir = os.path.join(self.output_dir, f'{model_name}-ct2-{option}')
            copy_files = [filename for filename in os.listdir(model_dir) if not filename.endswith(('.bin', '.safetensors', 'onnx', 'Pooling')) and filename not in ["config.json", ".git", "coreml", "configs", "runs", ".idea"]]
            if copy_files:
                copy_files_option = ' '.join([f'"{filename}"' for filename in copy_files])
                copy_files_cmd_part = f'--copy_files {copy_files_option} '
            else:
                copy_files_cmd_part = ''
            
            if option == "awq":
                cmd = (f'ct2-transformers-converter --model "{model_dir}" '
                       f'--output_dir "{output_dir}" '
                       f'--low_cpu_mem_usage '
                       f'--trust_remote_code '
                       f'{copy_files_cmd_part.strip()}')
            else:
                cmd = (f'ct2-transformers-converter --model "{model_dir}" '
                       f'--output_dir "{output_dir}" '
                       f'--quantization {option} '
                       f'--low_cpu_mem_usage '
                       f'--trust_remote_code '
                       f'{copy_files_cmd_part.strip()}')
            
            self.current_conversion = ConversionThread(cmd, option)
            self.current_conversion.started.connect(self.on_conversion_started)
            self.current_conversion.finished.connect(self.on_conversion_finished)
            self.current_conversion.error.connect(self.on_conversion_error)
            self.current_conversion.start()
        else:
            self.append_to_text_widget("All selected conversions are complete.")

    def on_conversion_started(self, quantization, command):
        self.append_to_text_widget(f"Starting conversion for {quantization} with command:\n{command}")

    def on_conversion_finished(self, quantization, output):
        completion_message = f"Conversion completed for {quantization}.\n{output}"
        self.append_to_text_widget(completion_message)
        self.process_next_conversion()

    def on_conversion_error(self, error_message):
        self.append_to_text_widget(error_message)

if __name__ == "__main__":
    app = QApplication([])
    app.setStyle(QStyleFactory.create("Fusion"))
    widget = App()
    widget.show()
    app.exec()
