from QRDetector import QRDetector
import cv2

if __name__ == '__main__':
    detector = QRDetector()
    image = cv2.imread('resources/test.jpg')
    detections = detector.detect(image)
    print(detections)