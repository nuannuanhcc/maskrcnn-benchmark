#export NGPUS=4

#rlaunch --positive-tags=1080ti --cpu=4 --gpu=4 --memory=$((1024*10)) --preemptible=no  --max-wait-time 5h \
# -- python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
# --config-file "configs/sysu_faster_R_50_FPN.yaml"

# rlaunch --positive-tags=1080ti --cpu=4 --gpu=4 --memory=$((1024*10)) --preemptible=no  --max-wait-time 5h \
# -- python ./tools/train_net.py \
# --config-file "configs/sysu_faster_R_50_FPN.yaml" \
# SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1



export NGPUS=$2

#rlaunch --positive-tags=1080ti --cpu=8 --gpu=$2 --memory=$((1024*30)) --preemptible=no  --max-wait-time 5h \
# -- python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
# --config-file $4 SOLVER.IMS_PER_BATCH $1 SOLVER.BASE_LR $3 TEST.IMS_PER_BATCH $2 SUBDIR $5

 python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py \
 --config-file $4 SOLVER.IMS_PER_BATCH $1 SOLVER.BASE_LR $3 TEST.IMS_PER_BATCH $2 SUBDIR $5




