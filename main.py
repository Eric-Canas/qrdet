from qrdet import QRDetector, _plot_result
import cv2

if __name__ == '__main__':
    detector = QRDetector()
    image = cv2.cvtColor(cv2.imread(filename='resources/qreader_test_image.jpeg'), code=cv2.COLOR_BGR2RGB)
    detections = detector.detect(image=image, is_bgr=False, legacy=False)
    _plot_result(image=image, detections=detections)
