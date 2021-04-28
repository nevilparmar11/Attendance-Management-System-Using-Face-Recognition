![Project Logo](https://user-images.githubusercontent.com/48133426/116291303-0a138200-a7b2-11eb-963c-5ead9628ec90.jpg)

## Attendance Management System Using Face Recognition 💻
[Link To Presentation](https://youtu.be/6qQZr9h8qL0)

This project involves building an attendance system which utilizes facial recognition to mark the presence, time-in, and time-out of employees. It covers areas such as facial detection, alignment, and recognition, along with the development of a web application to cater to various use cases of the system such as registration of new employees, addition of photos to the training dataset, viewing attendance reports, etc. This project intends to serve as an efficient substitute for traditional manual attendance systems. It can be used in corporate offices, schools, and organizations where security is essential.

This project aims to automate the traditional attendance system where the attendance is marked manually. It also enables an organization to maintain its records like in-time, out time, break time and attendance digitally. Digitalization of the system would also help in better visualization of the data using graphs to display the no. of employees present today, total work hours of each employee and their break time. Its added features serve as an efficient upgrade and replacement over the traditional attendance system.

## Scope of the project 🚀
Facial recognition is becoming more prominent in our society. It has made major progress in the field of security. It is a very effective tool that can help low enforcers to recognize criminals and software companies are leveraging the technology to help users access the technology. This technology can be further developed to be used in other avenues such as ATMs, accessing confidential files, or other sensitive materials.
This project servers as a foundation for future projects based on facial detection and recognition. This project also convers web development and database management with a user-friendly UI. Using this system any corporate offices, school and organization can replace their traditional way of maintaining attendance of the employees and can also generate their availability(presence) report throughout the month.

**The system mainly works around 2 types of users**
1. Employee
2. Admin

**Following functionalities can be performed by the admin: <br>**
• Login <br>
• Register new employees to the system <br>
• Add employee photos to the training data set <br>
• Train the model <br>
• View attendance report of all employees. Attendance can be filtered by date or employee. <br>

**Following functionalities can be performed by the employee: <br>**
• Login <br>
• Mark his/her time-in and time-out by scanning their face <br>
• View attendance report of self <br>

## Face Detection
Dlib's HOG facial detector.

## Facial Landmark Detection
Dlib's 68 point shape predictor

## Extraction of Facial Embeddings
face_recognition by Adam Geitgey

## Classification of Unknown Embedding 
using a Linear SVM (scikit-learn)

## Documentation 📰
[This](https://github.com/nevilparmar11/Attendance-Management-System-Using-Face-Recognition/tree/main/Documentation) folder contains all the related documents with UML Diagrams. 
 
## How To Run ?
- clone it on your computer
- make a separate [python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) or use the default one already installed on your machine
- Download [this](https://drive.google.com/uc?export=download&id=1HzO-rnEqgkZ6tLt48yWhYgHk1_zOIYhf) file 
 - put it inside **``` \Attendance-System-Using-Face-Recognition\face_recognition_data ```** directory
- run **``` pip install -r requirements.txt inside \Attendance-System-Using-Face-Recognition ```** directory
- Run **``` python manage.py runserver ```** inside **``` \Attendance-System-Using-Face-Recognition ```** directory to run the project
- Enjoy !

## UI 💻
<img src="https://user-images.githubusercontent.com/48133426/116291303-0a138200-a7b2-11eb-963c-5ead9628ec90.jpg" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116292978-f6691b00-a7b3-11eb-9d81-b787f7d7e790.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116292839-cae63080-a7b3-11eb-980e-85efe6094f8d.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116292873-d5a0c580-a7b3-11eb-936e-cf810f3af326.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116292881-d9344c80-a7b3-11eb-8cee-ca441422c14d.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116292903-e3564b00-a7b3-11eb-9df5-17d22cfb976a.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116292925-e7826880-a7b3-11eb-9657-b1401f9caa9a.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116293007-ff59ec80-a7b3-11eb-8668-591289a589a5.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116293052-0bde4500-a7b4-11eb-84c1-bbefd8985b3c.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116293066-113b8f80-a7b4-11eb-8578-3f58bb055efd.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116293080-1567ad00-a7b4-11eb-87e0-b72b929e61af.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116293113-21536f00-a7b4-11eb-9157-d076affd789d.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116293119-244e5f80-a7b4-11eb-85d1-7461e17f56c6.png" width="45%"></img> <img src="https://user-images.githubusercontent.com/48133426/116293707-c9693800-a7b4-11eb-8a22-217e89bb3a2f.png" width="45%"></img> 

## Presentation 🎓
[Link To Presentation](https://youtu.be/6qQZr9h8qL0)

---------

```javascript

if (youEnjoyed) {
    starThisRepository();
}

```

-----------

## Thank You
- Author : [Nevil Parmar](https://nevilparmar.me)
