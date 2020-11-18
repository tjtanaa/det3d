from easydict import EasyDict as edict
import numpy as np


__C = edict()
cfg = __C

# 0. basic config
__C.TAG = 'default'
__C.CLASSES = ['Car']
__C.CLASSES_ID_MAPPING = {
                "Background":        0,
                "Car":               1,
                "Pedestrian":        2,
                "Person_sitting":    2,
                "Cyclist":           3,
                "Truck":             4,
                "Van":               1,
                "Tram":              4,
                "Misc":              4,
                }

__C.ID_CLASSES_MAPPING = {
                "0": "Background",
                "1": "Car",
                "2": "Pedestrian",
                "3": "Cyclist",
                "4": "Misc",
                }

__C.INCLUDE_SIMILAR_TYPE = False

# config of augmentation
__C.AUG_DATA = True
__C.AUG_METHOD_LIST = ['scaling', 'flip']
# __C.AUG_METHOD_LIST = ['rotation', 'scaling', 'flip']
__C.AUG_METHOD_PROB = [0.5, 0.5, 0.5]
__C.AUG_ROT_RANGE = 18

__C.GT_AUG_ENABLED = True
__C.GT_EXTRA_NUM = 15
__C.GT_AUG_RAND_NUM = False
__C.GT_AUG_APPLY_PROB = 0.75
__C.GT_AUG_HARD_RATIO = 0.6

__C.PC_REDUCE_BY_RANGE = True
__C.PC_AREA_SCOPE = np.array([[-40, 40],
                              [-1,   3],
                              [0, 70.4]])  # x, y, z scope in rect camera coords

__C.CLS_MEAN_SIZE = np.array([[1.52, 1.63, 3.88]], dtype=np.float32)

# 1. config of rpn network
__C.RPN = edict()
__C.RPN.ENABLED = True
__C.RPN.FIXED = False

__C.RPN.USE_INTENSITY = True

# config of testing
__C.RPN.SCORE_THRESH = 0.3


# 2. config of rcnn network
__C.RCNN = edict()
__C.RCNN.ENABLED = False

# config of input
__C.RCNN.USE_RPN_FEATURES = True
__C.RCNN.USE_MASK = True
__C.RCNN.MASK_TYPE = 'seg'
__C.RCNN.USE_INTENSITY = False
__C.RCNN.USE_DEPTH = True
__C.RCNN.USE_SEG_SCORE = False
__C.RCNN.ROI_SAMPLE_JIT = False
__C.RCNN.ROI_FG_AUG_TIMES = 10

__C.RCNN.REG_AUG_METHOD = 'multiple'  # multiple, single, normal
__C.RCNN.POOL_EXTRA_WIDTH = 1.0

# config of testing
__C.RCNN.SCORE_THRESH = 0.3
__C.RCNN.NMS_THRESH = 0.1


__C.TEST = edict()
__C.TEST.SPLIT = 'val'
__C.TEST.RPN_PRE_NMS_TOP_N = 9000
__C.TEST.RPN_POST_NMS_TOP_N = 300
__C.TEST.RPN_NMS_THRESH = 0.7
__C.TEST.RPN_DISTANCE_BASED_PROPOSE = True


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(type(value), type(d[subkey]))
        d[subkey] = value


def save_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], edict):
            if logger is not None:
                logger.info('\n%s.%s = edict()' % (pre, key))
            else:
                print('\n%s.%s = edict()' % (pre, key))
            save_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue

        if logger is not None:
            logger.info('%s.%s: %s' % (pre, key, val))
        else:
            print('%s.%s: %s' % (pre, key, val))
