from qrdet import QRDetector
import cv2

if __name__ == '__main__':
    detector = QRDetector()
    image = cv2.imread(filename='resources/test_mobile.jpeg')
    detections = detector.detect(image=image, is_bgr=True)

    # Draw the detections
    for (x1, y1, x2, y2), confidence in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(image, f'{confidence:.2f}', (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 255, 0), thickness=1)
    # Save the results
    cv2.imwrite(filename='resources/test_mobile_detections.jpeg', img=image)
