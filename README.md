# Violence Detection Project

This project aims to detect violent scenes in videos using a pre-trained model. You can test the model locally or through an API to classify frames based on the probability of violence.

---

## 📂 Dataset

You can download the dataset used to train the model from the following Google Drive link:

- [Violence and Non-Violence Dataset - Google Drive](https://drive.google.com/drive/folders/1ajn53cAi9eTOP70BukibQB7Ta-cjjx1-?usp=drive_link)

Please ensure that the dataset is organized in the following directory structure within the project:


---

## 🧪 Model Testing

To test the model locally, follow these steps:

1. Prepare a video file that you want to analyze for violent scenes.
2. Use the testing script to process the video. The script analyzes the video and detects frames where violence occurs, showing the probability of violence for each detected frame.
3. The frames classified as violent will be saved in a specified directory, and they will be displayed with their probabilities.

### 📸 Example Test Result
*Include a screenshot here showing the output of the test, with frames where violence was detected along with their probabilities.*
![image](https://github.com/user-attachments/assets/f97ff77a-4d27-4d02-8bad-76d477aed8e0)

![image](https://github.com/user-attachments/assets/f516e554-4da1-494b-83f0-6d9877b3826d)

---

## 🌐 API Testing

You can also test the API by sending a video file and receiving frames where violence is detected, along with the probability of each frame. 

### API Testing Steps
1. Ensure your API is running.
2. Send a video file through a POST request to the API.
3. The API will return the frames that contain violence along with their probabilities.

### 📸 API Test Response Example
*Include a screenshot here showing the API response, displaying detected frames and their violence probabilities.*

![image](https://github.com/user-attachments/assets/2249b24c-b95a-4314-bf7e-689a51946502)


![image](https://github.com/user-attachments/assets/442c30a2-a694-4cc9-ba64-0122143cd314)

---


git add README.md
git commit -m "Updated README with Google Drive dataset link"
git push origin main

