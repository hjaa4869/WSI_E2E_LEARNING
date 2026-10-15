# CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir ./patches/camelyon16_256_level1 --data_slide_dir ./data_storage/camelyon/camelyon16/training --csv_path ./dataset_csv/camelyon16.csv --feat_dir ./data_feat/camelyon16_gigapath_224 --batch_size 16 --slide_ext .tif
#dct_process和pretrained需要手动设置了


#panda数据集
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir ./patches/panda_level0_256 --data_slide_dir ./data_storage/panda --csv_path ./dataset_csv/panda.csv --feat_dir ./data_feat/panda_uni --batch_size 128 --slide_ext .tiff