#python train.py --data_path /home/lampson/2T_disk/Data/FaceMask/FaceMaskDataset/classification/CropedVOCfilter --log_path log/0.01Adam_batch128_epoch200 --saved_path models/0.01Adam_batch128_epoch200  --lr 0.01 --epochs 200

lr=0.01
epochs=300
batch_size=1024

#CUDA_VISIBLE_DEVICES=7 python main.py --data_path /home/lampson/raid5/FaceMask/CropedVOCfilter_filter --log_path log/${lr}SGD_batch{batch_size}_epoch${epochs}_gpu1_normal --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu1_normal  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size}

#CUDA_VISIBLE_DEVICES=7 python train.py --data_path /home/lampson/raid5/CropedVOCfilter --log_path log/${lr}SGD_batch1024_epoch${epochs} --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size}

#CUDA_VISIBLE_DEVICES=5,6,7 python main.py --gpus 3 --data_path /home/lampson/raid5/CropedVOCfilter --log_path log/${lr}Adam_batch${batch_size}_epoch${epochs}_gpu3 --saved_path models/${lr}Adam_batch${batch_size}_epoch${epochs}_gpu3  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size}

#CUDA_VISIBLE_DEVICES=7 python main.py --gpus 1 --data_path /home/lampson/raid5/CropedVOCfilter --log_path log/${lr}Adam_batch${batch_size}_epoch${epochs}_gpu1 --saved_path models/${lr}Adam_batch${batch_size}_epoch${epochs}_gpu1  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size} --mixed_precision False

#CUDA_VISIBLE_DEVICES=7 python main_single.py --data_path /home/lampson/raid5/CropedVOCfilter --log_path log/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu1 --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu1  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size}

CUDA_VISIBLE_DEVICES=6,7 python main.py --gpus 2 --data_path /home/lampson/raid5/FaceMask/CropedVOCfilter_filter --log_path log/${lr}Adam_batch${batch_size}_epoch${epochs}_gpu2_CE --saved_path models/${lr}Adam_batch${batch_size}_epoch${epochs}_gpu2_CE  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size} --loss_type CE

#CUDA_VISIBLE_DEVICES=5,6,7 python main.py --gpus 3 --data_path /home/lampson/raid5/CropedVOCfilter --log_path log/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu3 --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu3  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size}

#CUDA_VISIBLE_DEVICES=5 python main.py --gpus 1 --data_path /home/lampson/raid5/CropedVOCfilter --log_path log/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu1_CE_weight1.05 --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu1_CE_weight1.05  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size} --loss_type CE

#CUDA_VISIBLE_DEVICES=6,7 python main.py --gpus 2 --data_path /home/lampson/raid5/CropedVOCfilter --log_path log/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu2 --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu2  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size}

#CUDA_VISIBLE_DEVICES=6,7 python main.py --gpus 2 --data_path /home/lampson/raid5/FaceMask/CropedVOC --log_path log/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu2_BCE_lessData --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu2_CE_lessData  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size} --loss_type BCE

#CUDA_VISIBLE_DEVICES=6,7 python main.py --gpus 2 --data_path /home/lampson/raid5/FaceMask/CropedVOCfilter_filter --log_path log/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu2_BCE_filter --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu2_BCE_filter  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size} --loss_type BCE

#CUDA_VISIBLE_DEVICES=3,5,6,7 python main.py --gpus 4 --data_path /home/lampson/raid5/FaceMask/CropedVOCfilter_filter --log_path log/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu4_BCE --saved_path models/${lr}SGD_batch${batch_size}_epoch${epochs}_gpu4_BCE  --lr ${lr} --epochs ${epochs} --batch_size ${batch_size}
