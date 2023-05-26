# Water Meter Recognition
Machine Learning Cource final project,
*Kai Wang, 21052222 at hdu dot edu dot com*

## Problem Description
![imgs](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/data.png)
#### read dights in the photo of water meter
## Problem Solution
### WORKFLOW:  Rotation Regression → Object Detection → Digits Recognition
### *1.Rotation Regression*
predict the rotation angle of images.

![rota_reg](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/rota_reg.png)

### *2.Object Detecion*
detect the region that contains object of digits.

![object_detect](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/detect.png)

### *3.Digits Recognition*
recognize digits in the region detected.

![region_seged](https://github.com/iaoqian/water_meter_recognition/blob/main/IMGS/train_seg_1.jpg)
# → '0 0 1 4 1' #

