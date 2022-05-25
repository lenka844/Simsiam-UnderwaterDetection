# CUDA_VISIBLE_DEVICES=2 python DDP_simsiam_ccrop.py configs/small/underwater/simsiam_ccrop.py
# cd ../my_tools
# python self-weight_converter.py
# cd ../mmdetection
# bash train.sh


CUDA_VISIBLE_DEVICES=1 python DDP_simsiam_ccrop.py configs/small/underwater/simsiam_ccrop_full.py
cd ../my_tools
python self-weight_converter.py
# cd ../mmdetection
# bash train.sh