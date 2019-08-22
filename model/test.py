import numpy as np
import tensorflow as tf

class _CateLoder:
    '''
    return:
    cates : [c, cate_len]
    mask : [c, cate_len]
    '''
    def __init__(self, file_name = 'model/cate_indx.txt', cate_length = 10):
        self.cate_length = cate_length
        self.file_name = file_name

    @property
    def __call__(self):
        cates = []
        with open(self.file_name, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                cates.append(line)
        cate_strs =[]
        masks = []
        for item in cates:
            temp = np.fromstring(item , sep=",", dtype=int).tolist()
            cate_init = [0]*10
            mask_init = [0]*10
            for i in range(len(temp)):
                cate_init[i] += temp[i]
                mask_init[i] += 1
            cate_strs.append(cate_init)
            masks.append(mask_init)
        cate_strs = tf.convert_to_tensor(cate_strs)
        masks = tf.convert_to_tensor(masks)
        return cate_strs, masks