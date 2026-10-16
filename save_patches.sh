CUDA_VISIBLE_DEVICES=0 python save_patches.py --data_h5_dir ./patches/dear_camelyon16_256_level2 --data_slide_dir ./data_storage/camelyon/camelyon16/training --csv_path ./dataset_csv/camelyon16.csv --patch_dir ./lmdb_feat --batch_size 1024 --slide_ext .tif --save_lmdb --img_size 256 --lmdb_name camelyon16_256_level2 --workers 6


#未设置保存小切片对img和lmdb，同时相关的图像质量参数也未在此脚本进行设置，但是那两个参数一般很少被用到，所以暂时忽视

# 使用时需要激活环境 conda activate abmilx