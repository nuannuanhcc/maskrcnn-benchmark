
#rlaunch --positive-tags=1080ti --cpu=4 --gpu=4 --memory=$((1024*10)) --preemptible=no  --max-wait-time 5h \

export NGPUS=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
 --config-file "configs/sysu_mask_R_50_FPN.yaml" \
 SOLVER.IMS_PER_BATCH 4 \
 SOLVER.BASE_LR 0.0001 \
 TEST.IMS_PER_BATCH 1 \
 SUBDIR 5.12_mask_18000 \
 MODEL.WEIGHT "/unsullied/sharefs/hanchuchu/isilon-home/train_log/5.12_mask_12000/GPU4_LR0.0003/model_0009000.pth" \
 REID.TEST.WEIGHT "/unsullied/sharefs/hanchuchu/isilon-home/train_log/5.9_reid_bn/GPU4_LR0.003/reid_model_0018000.pth"
 #MODEL.WEIGHT "/unsullied/sharefs/hanchuchu/isilon-home/train_log/mask/model_0005000.pth"





