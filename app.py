from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app) 

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

# Function to analyze the video and extract violent moments with their probabilities and frames
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    violent_moments = []  # Use a list to hold dictionaries for each moment
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    seconds_per_frame = 1 / fps
    skip_frames = int(12 * fps)  # Number of frames to skip for 10 seconds

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
            
            # Save the frame as an image
            image_filename = f"frame_{minutes}_{seconds:02d}.jpg"
            image_path = os.path.join('frames', image_filename)
            cv2.imwrite(image_path, frame)

            # Append a dictionary with 'time', 'violence', and 'image' keys
            violent_moments.append({
                "time": time_formatted,
                "violence": round(float(prediction), 2),  # Round to 2 decimal places
                "image": f"/frames/{image_filename}"  # Add the image path
            })

            # Skip the next 10 seconds worth of frames
            frame_count += skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            continue  # Skip to the next iteration

        frame_count += 1

    cap.release()
    return violent_moments

# Route to serve the saved images
@app.route('/frames/<filename>')
def serve_frame(filename):
    return send_from_directory('frames', filename)

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
    # Create necessary directories if they don't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('frames'):
        os.makedirs('frames')

    app.run(host='0.0.0.0', port=5000)
