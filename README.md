# Object-Detection-In-Low-light-Illumination
Capstone Project of Computer Vision Course
## Members
Nguyen Hoang Dang - 20194423

Do Quoc An - 20194414

Ha Vu Thanh Dat - 20194424

Le Hai Son - 20194449

## Enviroment
```
python 3.9
pytorch 1.10.0
mmcv 1.5.3
opencv-python tqdm
```
## Pre-process
**Step-1:** Cd in "your_project_path", and do set-up process (see [mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) if you want find details): 
```
git clone https://github.com/dang-nh/Object-Detection-In-Night-Vision.git
```
```
cd "your project path"
```
```
pip install -r requirements.txt
```
**Step-2:** Download checkpoint and config file from [goole drive](https://drive.google.com/drive/folders/1nz7RRy5v29nU_TwES50Z1BqfOwmtV1oi?usp=sharing)

**Step-3:** Put checkpoint and config file to "checkpoints" folder and "config" folder like below folder structure
```
├── configs
│   └── yolov3_only_image_process.py
├── checkpoints
│   └── best_mAP_epoch_43.pth
├── infer.py # inference code
├── train.py # training code
├── test.py # testing code
```
## Experiment Result

| class     | gts  | dets | recall | ap    |
|  ----  | ----  | ----  | ----  | ----  |
| Bicycle   | 212  | 737  | 0.910  | 0.825 |
| Boat      | 289  | 845  | 0.879  | 0.790 |
| Bottle    | 282  | 1029 | 0.876  | 0.763 |
| Bus       | 135  | 287  | 0.978  | 0.937 |
| Car       | 597  | 1537 | 0.931  | 0.853 |
| Cat       | 183  | 518  | 0.874  | 0.723 |
| Chair     | 466  | 1886 | 0.848  | 0.717 |
| Cup       | 366  | 966  | 0.869  | 0.794 |
| Dog       | 207  | 637  | 0.928  | 0.813 |
| Motorbike | 233  | 721  | 0.854  | 0.771 |
| People    | 1562 | 3860 | 0.908  | 0.825 |
| Table     | 333  | 1491 | 0.772  | 0.575 |
| mAP       |      |      |        | 0.782 |

## Inference
Infer model with a set of image
```
python3 infer.py --input_dir [input directory path] --output_dir [output directory path] --checkpoint checkpoints/best_mAP_epoch_43.pth --cfg configs/yolov3_only_image_process.py
```

**The code is largely borrow from mmdetection and unprocess, Thanks to their wonderful works**
