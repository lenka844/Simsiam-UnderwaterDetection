CUDA_VISIBLE_DEVICES=1 python DDP_simsiam_ccrop_pretrain.py configs/small/underwater/simsiam_ccrop_pretrain.py
cd ../my_tools
python self-weight_converter_pretrain.py
cd ../mmdetection
bash run.sh