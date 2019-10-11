
#ulimit -n 9082
#rlaunch --cpu=1 --gpu=1 --memory=$((1024*10)) --preemptible=no --positive-tags=1080ti --max-wait-time 5h -- \

python ./tools/test_net.py --config-file configs/sysu_faster_R_50_FPN.yaml \
 SUBDIR '10.10_detector' \
 MODEL.WEIGHT "/data/hanchuchu/datasets/person_search/trained_model/detection_123-GPU4_LR0.0025-model_0020000.pth"


# 'mask/model_0072500.pth'
# 'detection_123/model_0002500.pth'
# /unsullied/sharefs/hanchuchu/isilon-home/train_log/mask/e2e_mask_rcnn_R_50_FPN_1x.pth
# '4.25_mask_demo/GPU1_LR0.0001/model_0002500.pth'
# '5.4_mask_reg/GPU1_LR0.0001/model_0001500.pth'