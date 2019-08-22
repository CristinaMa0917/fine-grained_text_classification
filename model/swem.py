# -*- coding: UTF-8 -*-
from collections import namedtuple
from argparse import Namespace
import tensorflow as tf
from tensorflow.contrib import layers
from util import diagnose
import numpy as np

try:
    SparseTensor = tf.sparse.SparseTensor
    to_dense = tf.sparse.to_dense
except:
    SparseTensor = tf.SparseTensor
    to_dense = tf.sparse_tensor_to_dense

"""
This script implements SWEM according to the paper below:
Shen D, Wang G, Wang W, et al. Baseline needs more love: On simple word-embedding-based models and associated pooling mechanisms[J]. arXiv preprint arXiv:1805.09843, 2018.
"""

class _WordEmbeddingEncoder:
    def __init__(self, word_count, dimension, training, scope_name, *args, **kwargs):
        training = training
        self._scope_name = scope_name

        with tf.variable_scope(scope_name, reuse=False):
            self._word_embedding = tf.get_variable(
                "WordEmbedding",
                [word_count + 1, dimension],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=training
            )

    def __call__(self, tokens):
        """
        获得句子的embedding
        :param tokens: batch_size * max_seq_len
        :param masks: batch_size * max_seq_len
        :return:
        """
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            return tf.nn.embedding_lookup(self._word_embedding, tokens)

class _CateEmbeddingEncoder:
    def __init__(self, cate_count=12764, dimension=100, training=1, scope_name='cate-id-encoder', *args, **kwargs):
        training = training
        self._scope_name = scope_name

        with tf.variable_scope(scope_name, reuse=False):
            self._word_embedding = tf.get_variable(
                "WordEmbedding",
                [cate_count + 1, dimension],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=training
            )

    def __call__(self, tokens):
        """
        获得句子的embedding
        :param tokens: batch_size * max_seq_len
        :param masks: batch_size * max_seq_len
        :return:
        """
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            return tf.nn.embedding_lookup(self._word_embedding, tokens)

class _MaxPooling:
    def __init__(self, **kwargs):
        pass

    def __call__(self, embeddings, masks):
        # batch * length * 1
        multiplier = tf.expand_dims(masks, axis=-1)
        embeddings_max = tf.reduce_max(
            tf.multiply(multiplier, embeddings),
            axis=1
        )
        return embeddings_max


class _AveragePooling:
    def __init__(self, **kwargs):
        pass

    def __call__(self, embeddings, masks):
        # batch * length * 1
        multiplier = tf.expand_dims(masks, axis=-1)
        embeddings_sum = tf.reduce_sum(
            tf.multiply(multiplier, embeddings),
            axis=1
        )

        length = tf.expand_dims(tf.maximum(tf.reduce_sum(masks, axis=1), 1.0), axis=-1)

        embedding_avg = embeddings_sum / length
        return embedding_avg


class _ConcatPooling:
    def __init__(self, **kwargs):
        pass

    def __call__(self, embeddings, masks):
        # batch * length * 1
        multiplier = tf.expand_dims(masks, axis=-1)
        masked_embedding = tf.multiply(embeddings, multiplier)
        embeddings_sum = tf.reduce_sum(masked_embedding, axis=1)
        length = tf.expand_dims(tf.maximum(tf.reduce_sum(masks, axis=1), 1.0), axis=-1)
        embedding_avg = embeddings_sum / length
        embeddings_max = tf.reduce_max(masked_embedding, axis=1)
        embeddings = tf.concat([embedding_avg, embeddings_max], axis=1)
        print(embeddings.shape)
        return embeddings

class _MlpTransformer(object):
    def __init__(self, layers, dropout, training, scope_name):
        self._layers = layers
        self._training = training
        self._dropout = dropout
        self._scope = tf.variable_scope(scope_name)

    def __call__(self, input, cate_id_emb, cate_name_emb, cate_masks): # c,100; c, s, 100
        with self._scope as scope:
            values = input

            for i, n_units in enumerate(self._layers[:-1], 1):
                with tf.variable_scope("MlpLayer-%d" % i) as hidden_layer_scope:
                    values = layers.fully_connected(
                        values, num_outputs=n_units, activation_fn=tf.nn.tanh,
                        scope=hidden_layer_scope, reuse=tf.AUTO_REUSE
                    )
                if self._training and self._dropout > 0:
                    print("In training mode, use dropout")
                    values = tf.nn.dropout(values, keep_prob=1 - self._dropout)

            cate_name_emb = tf.reduce_max(tf.multiply(tf.expand_dims(tf.cast(cate_masks,tf.float32),-1), cate_name_emb), axis=1) # c,e
            cate_emb = tf.concat([cate_id_emb, cate_name_emb], axis = 1) # c,2e

            for i, n_units in enumerate(self._layers[:-1], 1):
                with tf.variable_scope("MlpLayer1-%d" % i) as hidden_layer_scope:
                    cate_emb = layers.fully_connected(
                        cate_emb, num_outputs=n_units, activation_fn=tf.nn.tanh,
                        scope=hidden_layer_scope, reuse=tf.AUTO_REUSE
                    )
                if self._training and self._dropout > 0:
                    print("In training mode, use dropout")
                    values = tf.nn.dropout(values, keep_prob=1 - self._dropout)

            bias = tf.get_variable(
                    name = 'b',
                    shape=self._layers[-1],
                    trainable = self._training,
                    initializer= tf.constant_initializer(0)
            )
            return tf.matmul(values, tf.transpose(cate_emb,[1,0]))+bias, 0

class _Model:
    def __init__(self, word_encoder, cate_encoder, pooling, dropout, layers, training, scope_name, focal_loss = True):
        self._scope = tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE)
        self._word_encoder = word_encoder
        self._cate_encoder = cate_encoder
        self._pooling = pooling
        self._mlp = _MlpTransformer(layers, dropout, training=training, scope_name=scope_name + "_" + "MLP")
        self._focal = focal_loss
        self.cates, self.cate_masks = self._cate_loader()
        self.cate_name_embs = self._word_encoder(self.cates)
        self.cate_id_embs = self._cate_encoder(np.arange(12764))


    def _focal_loss(self, logits, labels, gamma=3):
        logits = tf.nn.softmax(logits, -1)
        labels = tf.one_hot(labels, depth=logits.shape[1]) # [batch_size,num_classes]
        L = -labels * ((1 - logits) ** gamma) * tf.log(logits+1e-32)
        L = tf.reduce_sum(L, axis=1)
        loss = tf.reduce_mean(L)
        return loss

    def _cate_loader(self, file_name = 'model/cate_indx.txt', cate_length = 10):
        cates = []
        with open(file_name, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                cates.append(line)
        cate_strs = []
        masks = []
        for item in cates:
            cate_temp = np.fromstring(item, sep=",", dtype=int).tolist()
            cate_init = [0] * cate_length
            mask_init = [0] * cate_length
            for i in range(len(cate_temp)):
                cate_init[i] += cate_temp[i]
                mask_init[i] += 1
            cate_strs.append(cate_init)
            masks.append(mask_init)
        cate_strs = tf.convert_to_tensor(cate_strs)
        masks = tf.convert_to_tensor(masks)
        return cate_strs, masks

    def __call__(self, query, mask, labels):
        with self._scope:
            word_embs = self._word_encoder(query)
            sentence_embeddings = self._pooling(word_embs, mask)
            logits, features = self._mlp(
                            input=sentence_embeddings,
                            cate_id_emb = self.cate_id_embs,
                            cate_masks = self.cate_masks,
                            cate_name_emb = self.cate_name_embs)
            prob = tf.nn.softmax(logits, dim =1)
            prediction = tf.argmax(prob, dimension=1)

            loss = None
            if labels is not None:
                if self._focal :
                    loss = self._focal_loss(logits , labels, 5)
                else:
                    loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=logits,
                            labels=labels
                        ))
            return Namespace(
                logit=logits,
                confidence=prob,
                feature=features,
                loss=loss,
                prediction=prediction
            )


class SWEModel:
    ModelConfigs = namedtuple("ModelConfigs", ("pooling", "dropout", "classes",
                                               "hidden_layers", "init_with_w2v",
                                               "dim_word_embedding","word_count"))

    def __init__(self, model_configs, train_configs, predict_configs, run_configs):
        self._model_configs = model_configs
        self._train_configs = train_configs
        self._predict_configs = predict_configs
        self._run_configs = run_configs

    def _train(self, model_output, labels):
        # TODO: Add optimizer / reguliazer
        optimizer = tf.train.AdamOptimizer(learning_rate=self._train_configs.learning_rate)
        train_op = optimizer.minimize(model_output.loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=model_output.loss,
            train_op=train_op,
            training_hooks=[
                tf.train.LoggingTensorHook(
                    {
                        "loss": model_output.loss,
                        "accuracy": 100. * tf.reduce_mean(
                            tf.cast(tf.equal(tf.cast(model_output.prediction, tf.int32), tf.cast(labels, tf.int32)),
                                    tf.float32)),
                        "step": tf.train.get_global_step()
                    },
                    every_n_iter=100
                )
            ]
        )

    def _predict(self, model_output):
        outputs = dict(oneid=model_output.oneid)

        if self._predict_configs.output_embedding:
            outputs["feature"] = tf.reduce_join(
                tf.as_string(model_output.feature),
                axis=1,
                separator=" "
            )

        if self._predict_configs.output_confidence:
            outputs["confidence"] = model_output.confidence

        if self._predict_configs.output_prediction:
            outputs["prediction"] = model_output.prediction

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=outputs
        )

    def _evaluate(self, model_output, labels):
        # 二分类评估指标
        if self._model_configs.classes == 2:
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=model_output.loss,
                eval_metric_ops={
                    "accuracy": tf.metrics.accuracy(labels, model_output.prediction),
                    "precision": tf.metrics.precision(labels, model_output.prediction),
                    "recall": tf.metrics.recall(labels, model_output.prediction),
                    "auc": tf.metrics.auc(labels, model_output.probability)
                }
            )
        # 多分类评估指标
        else:
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=model_output.loss,
                eval_metric_ops={
                    "accuracy": tf.metrics.accuracy(labels, model_output.prediction),
                    "mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(labels,
                                                                                  model_output.prediction,
                                                                                  self._model_configs.classes)
                }
            )

    def _build_model(self, features, labels, mode):
        oneid = features['oneid']
        query = features['words']
        mask = features['masks']

        training = mode is tf.estimator.ModeKeys.TRAIN
        word_encoder = _WordEmbeddingEncoder(
            scope_name="encoder",
            word_count=self._model_configs.word_count,
            dimension=self._model_configs.dim_word_embedding,
            training=training,
        )
        cate_encoder = _CateEmbeddingEncoder(
            scope_name="c-encoder",
            training=training,
        )

        if self._model_configs.pooling == 'concat':
            word_pool = _ConcatPooling()
        elif self._model_configs.pooling == 'max':
            word_pool = _MaxPooling()
        elif self._model_configs.pooling == 'ave':
            word_pool = _AveragePooling()
        else:
            print('Not implemented pooling')

        model = _Model(
            word_encoder=word_encoder,
            cate_encoder=cate_encoder,
            pooling=word_pool,
            dropout=self._model_configs.dropout,
            layers=[int(n) for n in self._model_configs.hidden_layers.split(",")] + [self._model_configs.classes],
            training=training,
            scope_name="Classification"
        )

        model_output = model(
            query = query,
            mask = mask,
            labels = labels
        )
        model_output.oneid = oneid
        return model_output

    def model_fn(self, features, labels, mode):
        model_output = self._build_model(features, labels, mode)

        if mode is tf.estimator.ModeKeys.TRAIN:
            return self._train(model_output, labels)
        elif mode is tf.estimator.ModeKeys.PREDICT:
            return self._predict(model_output)
        elif mode is tf.estimator.ModeKeys.EVAL:
            return self._evaluate(model_output, labels)
