
#rlaunch --positive-tags=1080ti --cpu=4 --gpu=4 --memory=$((1024*10)) --preemptible=no  --max-wait-time 5h \

export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
 --config-file "configs/sysu_mask_R_50_FPN.yaml" \
 SOLVER.IMS_PER_BATCH 12 \
 SOLVER.BASE_LR 0.0001 \
 TEST.IMS_PER_BATCH 1 \
 SUBDIR 5.8_mask_reg1 \
 MODEL.WEIGHT "/unsullied/sharefs/hanchuchu/isilon-home/train_log/mask/e2e_mask_rcnn_R_50_FPN_1x.pth"
 #MODEL.WEIGHT "/unsullied/sharefs/hanchuchu/isilon-home/train_log/mask/model_0005000.pth"





