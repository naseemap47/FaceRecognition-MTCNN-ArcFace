# FaceRecognition-MTCNN-ArcFace
FaceRecognition with MTCNN using ArcFace

<p align="center">
  <img src='https://user-images.githubusercontent.com/88816150/187910639-ae68998b-5377-40b7-8faf-0206d05353ae.gif' alt="animated" />
</p>

## 🚀 New Update (27-01-2023)
- ### Liveness Model:
  - Liveness detector capable of spotting fake faces and performing anti-face spoofing in face recognition systems
  - Our FaceRecognition system initaly will check the faces are Fake or NOT
  - If its a Fake face it will give warnings
  - Otherwise it will go for Face-Recognition

### Clone this Repository
```
git clone https://github.com/naseemap47/FaceRecognition-MTCNN-ArcFace.git
cd FaceRecognition-MTCNN-ArcFace
```

### Install dependency
```
pip3 install -r requirement.txt
```

# Custom Face Recognition
You can use:<br> **Command Line<br> OR<br> Streamlit** Dashboard
## Streamlit Dashboard
### Install Streamlit
```
pip3 install streamlit
```
### RUN Streamlit
```
streamlit run app.py
```

## Command Line (Recommended)
### 1.Collect Data using Web-cam or RTSP

<details>
  <summary>Args</summary>
  
  `-i`, `--source`: RTSP link or webcam-id <br>
  `-n`, `--name`: name of the person <br>
  `-o`, `--save`: path to save dir <br>
  `-c`, `--conf`: min prediction conf (0<conf<1) <br>
  `-x`, `--number`: number of data wants to collect

</details>

**Example:**
```
python3 take_imgs.py --source 0 --name JoneSnow --save data --conf 0.8 --number 100
```
:book: **Note:** <br>
Repeate this process for all people, that we need to detect on CCTV, Web-cam or in Video.<br>
In side save Dir, contain folder with name of people. Inside that, it contain collected image data of respective people.<br>
**Structure of Save Dir:** <br>
```
├── data_dir
│   ├── person_1
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── person_2
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```

### 2.Normalize Collected Data
It will Normalize all data inside path to save Dir and save same as like Data Collected Dir

<details>
  <summary>Args</summary>
  
  `-i`, `--dataset`: path to dataset/dir <br>
  `-o`, `--save`: path to save dir

</details>

**Example:**
```
python3 norm_img.py --dataset data/ --save norm_data
```
**Structure of Normalized Data Dir:** <br>
```
├── norm_dir
│   ├── person_1
│   │   ├── 1_norm.jpg
│   │   ├── 2_norm.jpg
│   │   ├── ...
│   ├── person_2
│   │   ├── 1_norm.jpg
│   │   ├── 2_norm.jpg
│   │   ├── ...
.   .
.   .
```
### 3.Train a Model using Normalized Data

<details>
  <summary>Args</summary>
  
  `-i`, `--dataset`: path to Norm/dir <br>
  `-o`, `--save`: path to save .h5 model, eg: dir/model.h5 <br>
  `-l`, `--le`: path to label encoder <br>
  `-b`, `--batch_size`: batch Size for model training <br>
  `-e`, `--epochs`: Epochs for Model Training

</details>

**Example:**
```
python3 train.py --dataset norm_data/ --batch_size 16 --epochs 100
```

## Inference

<details>
  <summary>Args</summary>
  
  `-i`, `--source`: path to Video or webcam or image <br>
  `-m`, `--model`: path to saved .h5 model, eg: dir/model.h5 <br>
  `-c`, `--conf`: min prediction conf (0<conf<1) <br>
  `-lm`, `--liveness_model`: path to **liveness.model** <br>
  `--le`, `--label_encoder`: path to label encoder

</details>

### On Image 
**Example:**
```
python3 inference_img.py --source test/image.jpg --model models/model.h5 --conf 0.85 \
                     --liveness_model models/liveness.model --label_encoder models/le.pickle
```
**To Exit Window - Press Q-Key**

### On Video or Webcam
**Example:**
```
# Video (mp4, avi ..)
python3 inference.py --source test/video.mp4 --model models/model.h5 --conf 0.85 \
                     --liveness_model models/liveness.model --label_encoder models/le.pickle
```
```
# Webcam
python3 inference.py --source 0 --model models/model.h5 --conf 0.85 \
                     --liveness_model models/liveness.model --label_encoder models/le.pickle
```
**To Exit Window - Press Q-Key**
