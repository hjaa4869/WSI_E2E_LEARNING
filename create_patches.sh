# CUDA_VISIBLE_DEVICES=0 python create_patches_fp.py --source ./data_storage/camelyon/camelyon16/training --save_dir ./patches/dear_camelyon16_256_level2 --patch_size 256 --step_size 256 --patch_level 2 --preset camelyon16.csv --seg --patch


CUDA_VISIBLE_DEVICES=0 python create_patches_fp.py --source ./data_storage/panda --save_dir ./patches/panda_level0_256 --patch_size 256 --step_size 256 --patch_level 0 --preset panda.csv --seg --patch