import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QSlider,
    QGroupBox,
    QStatusBar
)

from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt


class ImageProcessingApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 1100, 700)

        # Store images
        self.original_image = None
        self.current_image = None

        self.create_gui()

    # ---------------------------------------------------------
    # CREATE GUI
    # ---------------------------------------------------------
    def create_gui(self):

        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # -----------------------------------------------------
        # TITLE
        # -----------------------------------------------------
        title = QLabel("IMAGE PROCESSING APP")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)

        title.setFont(title_font)

        main_layout.addWidget(title)

        subtitle = QLabel(
            "PyQt5 Image Processing Lab Task"
        )

        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle)

        # -----------------------------------------------------
        # IMAGE PREVIEW
        # -----------------------------------------------------
        self.image_label = QLabel()

        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(700, 450)

        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #eeeeee;
                border: 2px solid #555555;
                border-radius: 8px;
            }
        """)

        self.image_label.setText(
            "Open an image to start processing"
        )

        main_layout.addWidget(self.image_label)

        # -----------------------------------------------------
        # BUTTON AREA
        # -----------------------------------------------------
        button_layout = QHBoxLayout()

        # Open button
        open_button = QPushButton("📂 Open Image")
        open_button.clicked.connect(self.open_image)

        # Grayscale button
        gray_button = QPushButton("🎨 Grayscale")
        gray_button.clicked.connect(self.grayscale)

        # Blur button
        blur_button = QPushButton("💧 Blur")
        blur_button.clicked.connect(self.blur_image)

        # Edge button
        edge_button = QPushButton("🔎 Edge Detection")
        edge_button.clicked.connect(self.edge_detection)

        # Save button
        save_button = QPushButton("💾 Save Image")
        save_button.clicked.connect(self.save_image)

        # Reset button
        reset_button = QPushButton("↩ Reset")
        reset_button.clicked.connect(self.reset_image)

        buttons = [
            open_button,
            gray_button,
            blur_button,
            edge_button,
            save_button,
            reset_button
        ]

        for button in buttons:

            button.setMinimumHeight(45)

            button.setStyleSheet("""
                QPushButton {
                    background-color: #2d89ef;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px;
                    font-size: 14px;
                    font-weight: bold;
                }

                QPushButton:hover {
                    background-color: #1b5fa7;
                }

                QPushButton:pressed {
                    background-color: #154a80;
                }
            """)

            button_layout.addWidget(button)

        main_layout.addLayout(button_layout)

        # -----------------------------------------------------
        # BLUR CONTROL
        # -----------------------------------------------------
        blur_group = QGroupBox("Blur Intensity")

        blur_layout = QHBoxLayout()

        self.blur_slider = QSlider(Qt.Orientation.Horizontal)

        self.blur_slider.setMinimum(1)
        self.blur_slider.setMaximum(15)
        self.blur_slider.setValue(5)

        # Make sure values are odd
        self.blur_slider.valueChanged.connect(
            self.update_blur
        )

        blur_layout.addWidget(
            QLabel("Blur:")
        )

        blur_layout.addWidget(
            self.blur_slider
        )

        blur_group.setLayout(blur_layout)

        main_layout.addWidget(blur_group)

        # -----------------------------------------------------
        # STATUS BAR
        # -----------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Ready - Please open an image"
        )

    # ---------------------------------------------------------
    # OPEN IMAGE
    # ---------------------------------------------------------
    def open_image(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if file_name:

            image = cv2.imread(file_name)

            if image is None:

                QMessageBox.critical(
                    self,
                    "Error",
                    "Unable to open the selected image."
                )

                return

            self.original_image = image.copy()
            self.current_image = image.copy()

            self.display_image(
                self.current_image
            )

            self.status_bar.showMessage(
                "Image opened successfully"
            )

    # ---------------------------------------------------------
    # DISPLAY IMAGE
    # ---------------------------------------------------------
    def display_image(self, image):

        if image is None:
            return

        # Convert BGR to RGB
        if len(image.shape) == 2:

            height, width = image.shape

            bytes_per_line = width

            q_image = QImage(
                image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_Grayscale8
            )

        else:

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            height, width, channels = rgb_image.shape

            bytes_per_line = channels * width

            q_image = QImage(
                rgb_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(q_image)

        # Resize preview while keeping aspect ratio
        pixmap = pixmap.scaled(
            self.image_label.width() - 20,
            self.image_label.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(pixmap)

    # ---------------------------------------------------------
    # GRAYSCALE
    # ---------------------------------------------------------
    def grayscale(self):

        if self.current_image is None:

            self.show_no_image_message()

            return

        # Convert image to grayscale
        gray = cv2.cvtColor(
            self.current_image,
            cv2.COLOR_BGR2GRAY
        )

        self.current_image = gray

        self.display_image(
            self.current_image
        )

        self.status_bar.showMessage(
            "Grayscale applied"
        )

    # ---------------------------------------------------------
    # BLUR
    # ---------------------------------------------------------
    def blur_image(self):

        if self.current_image is None:

            self.show_no_image_message()

            return

        value = self.blur_slider.value()

        # GaussianBlur requires an odd number
        if value % 2 == 0:
            value += 1

        # Minimum kernel size
        if value < 3:
            value = 3

        self.current_image = cv2.GaussianBlur(
            self.current_image,
            (value, value),
            0
        )

        self.display_image(
            self.current_image
        )

        self.status_bar.showMessage(
            "Blur applied"
        )

    # ---------------------------------------------------------
    # LIVE BLUR PREVIEW
    # ---------------------------------------------------------
    def update_blur(self):

        if self.original_image is None:
            return

        value = self.blur_slider.value()

        if value % 2 == 0:
            value += 1

        if value < 3:
            value = 3

        blurred = cv2.GaussianBlur(
            self.original_image,
            (value, value),
            0
        )

        self.current_image = blurred

        self.display_image(
            self.current_image
        )

        self.status_bar.showMessage(
            f"Live blur preview - Kernel: {value} x {value}"
        )

    # ---------------------------------------------------------
    # EDGE DETECTION
    # ---------------------------------------------------------
    def edge_detection(self):

        if self.current_image is None:

            self.show_no_image_message()

            return

        # Convert to grayscale if required
        if len(self.current_image.shape) == 3:

            gray = cv2.cvtColor(
                self.current_image,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = self.current_image

        # Canny edge detection
        edges = cv2.Canny(
            gray,
            100,
            200
        )

        self.current_image = edges

        self.display_image(
            self.current_image
        )

        self.status_bar.showMessage(
            "Canny edge detection applied"
        )

    # ---------------------------------------------------------
    # SAVE IMAGE
    # ---------------------------------------------------------
    def save_image(self):

        if self.current_image is None:

            self.show_no_image_message()

            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Image (*.png);;JPG Image (*.jpg)"
        )

        if file_name:

            success = cv2.imwrite(
                file_name,
                self.current_image
            )

            if success:

                QMessageBox.information(
                    self,
                    "Success",
                    "Image saved successfully!"
                )

                self.status_bar.showMessage(
                    "Image saved successfully"
                )

            else:

                QMessageBox.critical(
                    self,
                    "Error",
                    "Could not save the image."
                )

    # ---------------------------------------------------------
    # RESET IMAGE
    # ---------------------------------------------------------
    def reset_image(self):

        if self.original_image is None:

            self.show_no_image_message()

            return

        self.current_image = (
            self.original_image.copy()
        )

        self.display_image(
            self.current_image
        )

        self.status_bar.showMessage(
            "Image reset to original"
        )

    # ---------------------------------------------------------
    # NO IMAGE MESSAGE
    # ---------------------------------------------------------
    def show_no_image_message(self):

        QMessageBox.warning(
            self,
            "No Image",
            "Please open an image first."
        )

    # ---------------------------------------------------------
    # WINDOW RESIZE
    # ---------------------------------------------------------
    def resizeEvent(self, event):

        super().resizeEvent(event)

        if self.current_image is not None:

            self.display_image(
                self.current_image
            )


# =============================================================
# MAIN PROGRAM
# =============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = ImageProcessingApp()

    window.show()

    sys.exit(
        app.exec_()
    )