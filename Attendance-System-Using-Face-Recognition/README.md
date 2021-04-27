# Attendance-System-Face-Recognition

This project is a POC web application demonstrating the use of facial recognition for marking attendance built as a part of my PS -1 internship at [ViitorCloud Technologies, Ahmedabad](https://viitorcloud.com/). It is a web application that can be used by the company to manage attendance of its employees.

## Functionality Supported
- Admin and Employee Login
- Admin : Register new employees.
- Admin : Add employee photos to the training dataset.
- Admin: Train the model.
- Admin: View attendance reports of all employees. Attendance can be filtered by date or employee. 
- Employee - View attendance reports of self.

## Built Using
- **OpenCV** - Open Source Computer Vision and Machine Learning software library
- **Dlib** - C++ Library containing Machine Learning Algorithms
- **face_recognition** by Adam Geitgey 
- **Django**- Python framework for web development.

### Face Detection
- Dlib's HOG facial detector.

### Facial Landmark Detection
- Dlib's 68 point shape predictor

### Extraction of Facial Embeddings
- face_recognition by Adam Geitgey

### Classification of Unknown Embedding 
- using a Linear SVM (scikit-learn)

The application was tested on data from 25 employees at ViitorCloud Technologies, Ahmedabad.

- [Link to presentation](https://docs.google.com/presentation/d/1Hdo-wKfn3PZxa3964XFmFtiSQEWDXZtQIsgS3v-sfIc/edit?usp=sharing)
- [Link to report](https://drive.google.com/file/d/126ut3WItK8LcodA6t_1_gY5J6ARcuQAZ/view?usp=sharing)
