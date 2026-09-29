import cv2
import numpy as np
import os
from typing import Any, Optional
from urllib.request import urlretrieve
from urllib.error import URLError


# ============================================================
# LAB 15 - REAL-TIME OBJECT DETECTION
# Digital Image Processing
#
# Techniques:
# 1. Haar Cascade Face Detection
# 2. HSV Color Thresholding
# 3. Contour Detection
# 4. Bounding Box Visualization
# ============================================================


# ------------------------------------------------------------
# 1. Load Haar Cascade Classifier
# ------------------------------------------------------------

cascade_filename = "haarcascade_frontalface_default.xml"


def _find_cascade_path():
    if os.path.exists(cascade_filename):
        return os.path.abspath(cascade_filename)

    haar_dir = None
    try:
        haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if isinstance(haar_dir, str):
            candidate = os.path.join(haar_dir, cascade_filename)
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass

    try:
        candidate = cv2.samples.findFile(cascade_filename, required=False)
        if candidate and os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    try:
        candidate = os.path.join(
            os.path.dirname(cv2.__file__),
            "data",
            "haarcascades",
            cascade_filename
        )
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    return None


def _download_cascade():
    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "data/haarcascades/haarcascade_frontalface_default.xml"
    )
    try:
        print(f"Downloading Haar cascade from {url}...")
        urlretrieve(url, cascade_filename)
        return os.path.abspath(cascade_filename)
    except URLError as err:
        print("WARNING: Could not download Haar cascade:", err)
        return None


def _create_cascade_classifier(*args, **kwargs):
    classifier = getattr(cv2, "CascadeClassifier", None)
    if callable(classifier):
        try:
            return classifier(*args, **kwargs)
        except Exception:
            return None
    return None


def _load_cascade(cascade_path) -> Optional[Any]:
    if cascade_path is None:
        return None

    try:
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is None or not callable(cascade_cls):
            print("ERROR: cv2 does not expose CascadeClassifier in this build.")
            return None

        face_cascade: Any = cascade_cls(cascade_path)

        if hasattr(face_cascade, "empty") and face_cascade.empty():
            print("ERROR: Could not load face cascade.")
            return None

        print("Face cascade loaded successfully.")
        return face_cascade

    except Exception as e:
        print(f"ERROR loading face cascade: {e}")
        return None


cascade_path = _find_cascade_path()
if cascade_path is None:
    cascade_path = _download_cascade()

if cascade_path is not None:
    print(f"Haar cascade path: {cascade_path}")

face_cascade = _load_cascade(cascade_path)

if face_cascade is None:
    print("WARNING: Haar Cascade could not be loaded. Face detection disabled.")


# ------------------------------------------------------------
# 2. Start Webcam
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    print("Try changing VideoCapture(0) to VideoCapture(1).")
    exit()


print("================================================")
print(" Real-Time Object Detection Started")
print("================================================")
print("RED   -> Red Object")
print("GREEN -> Green Object")
print("BLUE  -> Blue Object")
print("FACE  -> Face")
print("Press Q to quit.")
print("================================================")


# ------------------------------------------------------------
# 3. Function for Color Object Detection
# ------------------------------------------------------------

def detect_color_object(frame, hsv, lower, upper, label, box_color):
    """
    Detect objects of a specific color using HSV thresholding.
    Draws bounding boxes around detected objects.
    """

    # Create binary mask
    mask = cv2.inRange(hsv, lower, upper)

    # Noise removal
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Process every detected contour
    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small objects/noise
        if area < 800:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )

        # Draw label background
        cv2.rectangle(
            frame,
            (x, y - 30),
            (x + w, y),
            box_color,
            -1
        )

        # Draw label
        cv2.putText(
            frame,
            label,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return mask


# ------------------------------------------------------------
# 4. Main Real-Time Processing Loop
# ------------------------------------------------------------

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror webcam image
    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # A. Face Detection using Haar Cascade
    # --------------------------------------------------------

    if face_cascade is not None:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Draw face bounding boxes
        for (x, y, w, h) in faces:

            # Blue bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3
            )

            # Label background
            cv2.rectangle(
                frame,
                (x, y - 35),
                (x + w, y),
                (255, 0, 0),
                -1
            )

            # Face label
            cv2.putText(
                frame,
                "Face",
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
    else:
        cv2.putText(
            frame,
            "Face detection unavailable",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


    # --------------------------------------------------------
    # B. Convert BGR image to HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # --------------------------------------------------------
    # C. RED COLOR DETECTION
    # --------------------------------------------------------
    #
    # Red wraps around the HSV hue range, therefore two
    # ranges are required.
    # --------------------------------------------------------

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    red_mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    red_mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    red_mask = cv2.bitwise_or(
        red_mask1,
        red_mask2
    )

    kernel = np.ones((5, 5), np.uint8)

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 800:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Red bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "Red Object",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------------
    # D. GREEN COLOR DETECTION
    # --------------------------------------------------------

    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])

    detect_color_object(
        frame,
        hsv,
        lower_green,
        upper_green,
        "Green Object",
        (0, 255, 0)
    )


    # --------------------------------------------------------
    # E. BLUE COLOR DETECTION
    # --------------------------------------------------------

    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([135, 255, 255])

    detect_color_object(
        frame,
        hsv,
        lower_blue,
        upper_blue,
        "Blue Object",
        (255, 0, 0)
    )


    # --------------------------------------------------------
    # F. Display Information
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Real-Time Object Detection",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # G. Show Result
    # --------------------------------------------------------

    cv2.imshow(
        "DIP Lab 15 - Object Detection",
        frame
    )


    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# ------------------------------------------------------------
# 5. Release Resources
# ------------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Program terminated successfully.")