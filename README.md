# YOLOv3
Keras(TF backend) implementation of yolo v3 objects detection. 

According to the paper [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

## Requirement
- OpenCV 3.4
- Python 3.6    
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3

## Quick start

- Download official [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and put it on top floder of project.

- Run the follow command to convert darknet weight file to keras h5 file. The `yad2k.py` was modified from [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).
```
python yad2k.py cfg\yolo.cfg yolov3.weights data\yolo.h5
```

- run follow command to show the demo. The result can be found in `images\res\` floder.
```
python demo.py
```

## Demo result

It can be seen that yolo v3 has a better classification ability than yolo v2.

<img width="400" height="350" src="/images/res/dog.jpg"/><img width="400" height="350" src="/images/res/person.jpg"/>

## TODO

- Train the model.

## Reference

	@article{YOLOv3,  
	  title={YOLOv3: An Incremental Improvement},  
	  author={J Redmon, A Farhadi },
	  year={2018}



## Copyright
See [LICENSE](LICENSE) for details.
