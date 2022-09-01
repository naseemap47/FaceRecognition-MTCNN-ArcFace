# FaceRecognition-RetinaFace-ArcFace
FaceRecognition with MTCNN using ArcFace

### Clone this Repository
```
git clone https://github.com/naseemap47/FaceRecognition-MTCNN-ArcFace.git
cd FaceRecognition-MTCNN-ArcFace
```

### Install dependency
```
pip3 install -r requirement.txt
```

## Custom Face Recognition
### 1.Collect Data using Web-cam
```
python3 take_imgs.py --name <name of person> --save <path to save dir>
```
**Example:**
```
python3 take_imgs.py --name JoneSnow --save data
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
│   ├── person_1
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```

### 2.Normalize Collected Data
It will Normalize all data inside path to save Dir and save same as like Data Collected Dir
```
python3 norm_img.py --dataset <path to collected data> --save <path to save Dir>
```
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
│   ├── person_1
│   │   ├── 1_norm.jpg
│   │   ├── 2_norm.jpg
│   │   ├── ...
.   .
.   .
```
### 3.Train a Model using Normalized Data
```
python3 train.py --dataset <path to normalized Data> --save <path to save model.h5>
```
**Example:**
```
python3 train.py --dataset norm_data/ --save model.h5
```

## Inference
### :book: Note: <br>
Open **inference_img.py** and **inference.py**:- <br>
Change **class_names** List into your class names. **Don't** forget to give in same order used for Training the Model. 
### On Image 
```
python3 inference_img.py --image <path to image> --model <path to model.h5> --conf <min model prediction confidence>
```
**Example:**
```
python3 inference_img.py --image data/JoneSnow/54.jpg --model model.h5 --conf 0.85
```
**To Exit Window - Press Q-Key**

### On Video or Webcam
```
python3 inference.py --source <path to video or webcam index> --model <path to model.h5> --conf <min prediction confi>
```
**Example:**
```
# Video (mp4, avi ..)
python3 inference_img.py --source test/video.mp4 --model model.h5 --conf 0.85
```
```
# Webcam
python3 inference_img.py --source 0 --model model.h5 --conf 0.85
```
**To Exit Window - Press Q-Key**
