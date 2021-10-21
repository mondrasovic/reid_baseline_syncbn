@echo off

set datasets_dir_path=E:/datasets

python create_uadetrac_reid_dataset.py %datasets_dir_path%/UA-DETRAC %datasets_dir_path%/UA-DETRAC_ReID/image_train -s 777 -S 89991 -c 0.2
