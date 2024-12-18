- Wrote a class in utils/prepare.py to prepare students_faces for training student classification model 

```
# To prepare student faces for training classification model
inst = prepFaces('path to students directory','destination path to prepare data','yolo model class which gets face')
inst.prepare() # prepares data
```


- Then to train faces we used yolov8-cls weights form *models*. The training of student faces is done in *facetrain.ipynb*. After training of classification model it's weights are saved in *student_faces*

- The weights of *student classification* models are loaded and inference on images is ran in *pipe.ipynb*


### Plan
- So the output given by the function in pipe is: bounding box co-ordinates (centred), class label, confidence score of the class

- If the confidence score of the class is more than 95% we draw the box around that face anthe.



- Used yolov11n_face froms *models* to extract the faces of students. 