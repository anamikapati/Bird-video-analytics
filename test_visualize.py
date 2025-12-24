import cv2
from detector import BirdDetector

VIDEO_PATH = "sample_videos/birds.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if not ret:
    print("❌ Failed to read video frame")
    cap.release()
    exit()

detector = BirdDetector()
detections = detector.detect(frame)

# Draw bounding boxes
for det in detections:
    x1, y1, x2, y2, conf = det
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Bird? {conf:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

cap.release()

# Show image
cv2.imshow("Detection Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("outputs/detection_test.jpg", frame)
print("✅ Saved detection image to outputs/detection_test.jpg")
