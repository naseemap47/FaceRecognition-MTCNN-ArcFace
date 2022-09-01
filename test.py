from deepface import DeepFace

result = DeepFace.verify(img1_path = "data/Aasish/2.jpg", 
                        img2_path = "data/Naseem/54.jpg", 
                        model_name='ArcFace',
                        enforce_detection=False
                        )
print(result)
