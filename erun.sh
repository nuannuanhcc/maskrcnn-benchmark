


rlaunch --positive-tags=1080ti --cpu=2 --gpu=1 --memory=$((1024*10)) --preemptible=no  --max-wait-time 5h \
 -- python ./tools/test_net.py --config-file configs/sysu_faster_R_50_FPN.yaml SUBDIR 'mask/model_0072500.pth'
 #'4.11_detection/GPU8_LR0.02/model_0002500.pth'



