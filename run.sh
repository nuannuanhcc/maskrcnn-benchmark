
#rlaunch --positive-tags=1080ti --cpu=4 --gpu=4 --memory=$((1024*10)) --preemptible=no  --max-wait-time 5h \

export NGPUS=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
 --config-file "configs/sysu_faster_R_50_FPN.yaml" \
 SOLVER.IMS_PER_BATCH 4 \
 SOLVER.BASE_LR 0.0001 \
 TEST.IMS_PER_BATCH 1 \
 SUBDIR 5.12_mask_18000 \
 MODEL.WEIGHT "/unsullied/sharefs/hanchuchu/isilon-home/train_log/6.21_detector/GPU1_LR0.0025/model_0001000.pth" \
 REID.TEST.WEIGHT "/unsullied/sharefs/hanchuchu/isilon-home/train_log/mask/resnet50_model_150.pth"
 #MODEL.WEIGHT "/unsullied/sharefs/hanchuchu/isilon-home/train_log/mask/model_0005000.pth"





