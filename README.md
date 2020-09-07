**Train your model** 

```
python main.py --data_path /home/lampson/2T_disk/Data/FaceMask/FaceMaskClassification/CropedVOCfilter_filter --lr 0.1 --gpus 1 --loss softmax --batch_size 24 --path test
```

revise the log path and saved path as your own

* **python 3.7**
* **pytorch 1.4**
* **opencv (cv2)**
* **pthflops**
* **torchsummary**


### Usage
- python infer.py --model ./models/0.1Adam_batch1024_epoch150_gpu2_CE_StrongLight/f1_best_checkpoint.pth --video /home/lampson/2T_disk/Data/images/mask_test_videos/warperror.m4v --v 2 --loss sigmoid
