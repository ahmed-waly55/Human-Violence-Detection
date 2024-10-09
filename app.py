from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the trained model in TFLite format
interpreter = tf.lite.Interpreter(model_path='violence_detection_model_quantized.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize the frame to (224, 224)
    frame = frame.astype(np.float32)  # Convert the frame to float32 type
    frame = np.expand_dims(frame, axis=0)  # Add an additional dimension
    frame = frame / 255.0  # Rescale values to [0, 1]
    return frame

# Function to analyze the video and extract violent moments in "minute:second" format
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    violent_moments = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    seconds_per_frame = 1 / fps
    skip_frames = int(10 * fps)  # Number of frames to skip for 10 seconds

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)

        interpreter.set_tensor(input_details[0]['index'], processed_frame)

        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        if prediction > 0.7:  # If the value > 0.7, classify as violence
            total_seconds = int(frame_count * seconds_per_frame)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            time_formatted = f"{minutes}:{seconds:02d}"  # Format as "minute:second"
            if time_formatted not in violent_moments:
                violent_moments.append(time_formatted)

            # Skip the next 10 seconds worth of frames
            frame_count += skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            continue  # Skip to the next iteration

        frame_count += 1

    cap.release()
    return violent_moments

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    video_file = request.files['video']
    video_path = f"./uploads/{video_file.filename}"
    video_file.save(video_path)

    violent_moments = analyze_video(video_path)
    return jsonify({"violent_moments": violent_moments})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(host='0.0.0.0', port=5000)
