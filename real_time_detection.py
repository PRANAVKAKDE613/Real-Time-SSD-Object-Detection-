import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# COCO 90 class labels (index starts at 1)
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
    9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella',
    31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
    87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

# Load the SSD model from TF Hub
print("Loading model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded.")

# Open webcam (0 = default cam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize + normalize frame
    img = cv2.resize(frame, (320, 320))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([img_rgb], dtype=tf.uint8)

    # Run detection
    outputs = detector(input_tensor)

    # Unpack outputs
    boxes = outputs['detection_boxes'][0].numpy()
    class_ids = outputs['detection_classes'][0].numpy().astype(np.int32)
    scores = outputs['detection_scores'][0].numpy()

    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] < 0.5:
            continue
        box = boxes[i]
        y1, x1, y2, x2 = box
        left, top, right, bottom = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = COCO_LABELS.get(class_ids[i], f"ID {class_ids[i]}")
        cv2.putText(frame, f"{label}: {int(scores[i]*100)}%", (left, top - 10),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
