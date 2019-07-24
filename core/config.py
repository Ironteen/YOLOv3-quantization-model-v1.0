from easydict import EasyDict as edict 

cfg   = edict()

cfg.IMAGE_WIDTH                               = 224
cfg.IMAGE_HEIGHT                              = 224 

cfg.WEIGHT_DECAY                              = 0.0005
cfg.BATCH_NORM_EPSILON                        = 1e-5
cfg.LEARN_RATE                                = 1e-5
cfg.DECAY                                     = 0.9
cfg.STDDEV                                    = 0.001
cfg.ALPHA                                     = 0.1
cfg.QUANT                                     = True
cfg.COCO_CLASSES                              = 80
cfg.COCO_STRIDES                              = [8, 16, 32]

# dataset path
cfg.IMAGENET_PATH                             = '/data/dataset/ILSVRC2012'
cfg.VOC_PATH                                  = '/data/dataset/VOC'

cfg.COCO_ANCHORS                              = './data/plusAI/basline_anchors.txt'
cfg.COCO_NAMES                                = './data/plusAI/coco.names'
cfg.LAYERS_NAME                               = './data/plusAI/layer_name.txt'

cfg.LOGDIR                                    = './log/summary'

cfg.SAVED_WEIGHT                              = './log/checkpoint'

cfg.PARAM_SAVE_PATH                           = './log/params'

cfg.KEEP_PROB                                 = 0.7
cfg.BATCH_SIZE                                = 28
cfg.MAX_EPOCH_NUM                             = 10000