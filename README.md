# Water Meter Recognition
Machine Learning Cource, Spring 2023, Final Project.
*Kai Wang, 21052222 at hdu dot edu dot com*
## At the Begining
This project is NOT aiming to show you some brilliant scores it achieved, instead, it tries to make you how I manage and solve a relatively complicated Machine Learning problem in real world.
As a matter of fact, there have been some blogs/guidance show you how to deal with this problem. Using semantic(image) segmentation + classification is a universal way to handle problems like this.
But I did it, for this specific task, in my way. Well, it's sure, when you keep on reading you'll find its limitaitions. But that's not actual my purpose.

## Problem Description
![imgs](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/data.png)
#### Read dights in the photo of water meter
## Problem Solution
### WORKFLOW:  Rotation Regression → Object Detection → Digits Recognition
### *1.Rotation Regression*
Predict the rotation angle of images.

![rota_reg](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/rota_reg.png)

### *2.Object Detecion*
Detect the region that contains object of digits.

![object_detect](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/detect.png)

### *3.Digits Recognition*
Recognize digits in the region detected.

![region_seged](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/train_seg_1.jpg)
# → '0 0 1 4 1' #

## TO MAKE IT BETTER:
It got a score of 0.818 in metric $\frac{TP + NP}{P + N}, i.e. Accuracy$, in test dateset. Generally speaking, not bad.
 
![submit_socre](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/submit_score.png)
 
But it surely can be improved and get a point of 0.9 or so, easily, I think.

Here are some facts and recomendations: 
1. In this project, I just employed some basic models, tricks and didn't modify those hyper-parameters cautiously. You can improve it.
2. By checking ./ProcessedData in WaterMeterDataset, you can find that rotation regression and object detection parts of the job were done well. Actaully, the bottleneck was right the digits recognition part.
3. Sligtly expand and segment the region detected in origin image may help. We would rather take in more noise than lose vital information.
