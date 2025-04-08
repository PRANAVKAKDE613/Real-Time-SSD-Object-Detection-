import tensorflow as tf
import tensorflow_hub as hub

# Load SSD MobileNet V2 from TF Hub
print("Loading model from TensorFlow Hub...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded successfully.")
