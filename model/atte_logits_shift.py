import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

class _AtteModule:
    '''
    word_embeddings :[b, s1, e]
    word_masks : [b,s1]
    cate_embeddings :[c, s2, e]
    cate_masks : [b,s2]
    '''
    def __init__(self, cate_embeddings, cate_masks, kernel_size, scope_name):
        masks = tf.expand_dims(cate_masks, axis=-1)
        emb_max = tf.reduce_max(tf.multiply(masks, cate_embeddings),axis = 1)
        self.cate_emb_norm = tf.nn.l2_normalize(emb_max, axis =1) # b,e
        self.scope = scope_name
        self.kernel_size = kernel_size

    def _partialWeights(self, word_embeddings, word_masks):
        masks = tf.expand_dims(word_masks, axis=-1)
        word_emb_max = tf.reduce_max(tf.multiply(masks, word_embeddings), axis = 1) # b,e
        with tf.variable_scope("MlpLayer-%s" % self.scope) as partial_weights_scope:
            partial_weights = layers.fully_connected(
                word_emb_max, num_outputs=1, activation_fn=tf.nn.tanh,
                scope=partial_weights_scope, reuse=tf.AUTO_REUSE
            )
        return partial_weights

    def _attention_layer(self, word_embeddings, word_masks):
        with tf.variable_scope("AttenLayer-%s" % self.scope) as atten_layer_scope:
            masks = tf.expand_dims(word_masks, axis = -1)
            word_emb = tf.multiply(masks,word_embeddings)
            word_emb_norm = tf.nn.l2_normalize(word_emb, axis =2)  # b * s1 * e
            G = tf.contrib.keras.backend.dot(word_emb_norm, tf.transpose(self.cate_emb_norm,[1,0]))  # b * s * c


            kernel_params = tf.get_variable(
                name = "Conv2d",
                shape = [self.kernel_size, self.cate_emb_norm.shape[0], 1, self.cate_emb_norm.shape[0]],
                dtype = tf.float32,
                initializer= tf.contrib.layers.xavier_initializer()
            )
            kernel_bias = tf.get_variable(
                name = "Conv2dBias",
                shape = [self.cate_emb_norm.shape[0]],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            att_v = tf.nn.conv2d(tf.expand_dims(G,axis =-1), kernel_params, strides=[1]*4, padding='VALID') # b,s-kc,1,c
            att_v = tf.nn.relu(tf.nn.bias_add(att_v, kernel_bias))
            att_v = tf.squeeze(att_v, 2) # b,s-kc,c
            att_v = tf.reduce_max(att_v , axis=1) # b, c
            print('att_v.shape',att_v.shape)
            att_logits = tf.nn.softmax(att_v, axis = -1) #b,c
            return att_logits


    def __call__(self, word_embeddings, word_masks):
        alpha = self._partialWeights(word_embeddings, word_masks)
        att_logits = self._attention_layer(word_embeddings, word_masks)
        return alpha, att_logits
