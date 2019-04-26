from .reid_baseline import REID_Baseline


def build_reid_model(cfg):
    if cfg.REID.MODEL.NAME == 'resnet50':
        model = REID_Baseline(cfg)
    return model



