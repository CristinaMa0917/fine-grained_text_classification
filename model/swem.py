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
    def __init__(self, layers, dropout, training, kid_count, scope_name):
        self._layers = layers
        self._training = training
        self._dropout = dropout
        self._scope = tf.variable_scope(scope_name)
        self._parent_index, self.par_count = self._cate_relation_loader()
        self.kid_count = kid_count

    def _cate_relation_loader(self, file_name='model/par_index.txt'):
        parent_index = []
        with open(file_name, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                parent_index.append(int(line))
        par_count = max(parent_index)+1
        return parent_index, par_count

    def __call__(self, input): # c,100;
        with self._scope as scope:
            values = input
            for i, n_units in enumerate(self._layers, 1):
                with tf.variable_scope("ParMlpLayer-%d" % i) as hidden_layer_scope:
                    values = layers.fully_connected(
                        values, num_outputs=n_units, activation_fn=tf.nn.tanh,
                        scope=hidden_layer_scope, reuse=tf.AUTO_REUSE
                    )
                if self._training and self._dropout > 0:
                    print("In training mode, use dropout")
                    values = tf.nn.dropout(values, keep_prob=1 - self._dropout)
            logits_par = layers.linear(values, self.par_count, scope="f-par-{}".format(i), reuse=tf.AUTO_REUSE)

            values = input
            for i, n_units in enumerate(self._layers, 1):
                with tf.variable_scope("KidMlpLayer1-%d" % i) as hidden_layer_scope:
                    values = layers.fully_connected(
                        values, num_outputs=n_units, activation_fn=tf.nn.tanh,
                        scope=hidden_layer_scope, reuse=tf.AUTO_REUSE
                    )
                if self._training and self._dropout > 0:
                    print("In training mode, use dropout")
                    values = tf.nn.dropout(values, keep_prob=1 - self._dropout)
            logits_kid_pre = layers.linear(values, self.kid_count, scope="f-kid-{}".format(i), reuse=tf.AUTO_REUSE)

            logits_par_transpose = tf.transpose(logits_par, [1, 0])
            logits_kid_from_par = tf.nn.embedding_lookup(logits_par_transpose, self._parent_index)
            logits_kid_from_par_trans = tf.transpose(logits_kid_from_par, [1, 0])
            logits_kid = logits_kid_from_par_trans + logits_kid_pre
            return logits_par, logits_kid

class _Model:
    def __init__(self, word_encoder, pooling, dropout, layers, training, scope_name, kid_count,  focal_loss = True):
        self._scope = tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE)
        self._word_encoder = word_encoder
        self._pooling = pooling
        self._mlp = _MlpTransformer(layers, dropout, kid_count=kid_count,  training=training, scope_name=scope_name + "_" + "MLP")
        self._focal = focal_loss

    def _focal_loss(self, logits, labels, gamma=3):
        logits = tf.nn.softmax(logits, -1)
        labels = tf.one_hot(labels, depth=logits.shape[1]) # [batch_size,num_classes]
        L = -labels * ((1 - logits) ** gamma) * tf.log(logits+1e-32)
        L = tf.reduce_sum(L, axis=1)
        loss = tf.reduce_mean(L)
        return loss

    def __call__(self, query, mask, kid_labels, par_labels):
        with self._scope:
            word_embs = self._word_encoder(query)
            sentence_embeddings = self._pooling(word_embs, mask)
            logits_par, logits_kid = self._mlp(input=sentence_embeddings)
            prob_kid = tf.nn.softmax(logits_kid, dim = 1)
            pred_kid = tf.argmax(prob_kid, dimension = 1)
            prob_par = tf.nn.softmax(logits_par, dim = 1)
            pred_par = tf.argmax(prob_par, dimension = 1)

            loss = None
            loss1 = None
            loss2 = None
            if kid_labels is not None:
                if self._focal :
                    loss1 = self._focal_loss(logits_par, par_labels, 5)
                    loss2 = self._focal_loss(logits_kid, kid_labels, 5)
                    loss = loss1 + loss2
                else:
                    loss1 = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=logits_par,
                            labels=par_labels
                        ))
                    loss2 = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=logits_kid,
                            labels=kid_labels
                        ))
                    loss = loss1 + loss2

            return Namespace(
                logit=logits_kid,
                confidence=prob_kid,
                feature=0,
                loss=loss,
                loss_par=loss1,
                loss_kid=loss2,
                prediction_kid=pred_kid,
                prediction_par=pred_par
            )

class SWEModel:
    ModelConfigs = namedtuple("ModelConfigs", ("pooling", "dropout",
                                               "hidden_layers", "init_with_w2v",
                                               "dim_word_embedding","word_count",
                                               "kid_count"))

    def __init__(self, model_configs, train_configs, predict_configs, run_configs):
        self._model_configs = model_configs
        self._train_configs = train_configs
        self._predict_configs = predict_configs
        self._run_configs = run_configs

    def _train(self, model_output, kid_labels, par_labels):
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
                        "kid_acc": 100. * tf.reduce_mean(
                            tf.cast(tf.equal(tf.cast(model_output.prediction_kid, tf.int32), tf.cast(kid_labels, tf.int32)),
                                    tf.float32)),
                        "par_acc": 100. * tf.reduce_mean(
                            tf.cast(tf.equal(tf.cast(model_output.prediction_par, tf.int32), tf.cast(par_labels, tf.int32)),
                                    tf.float32)),
                        "kid_loss":model_output.loss_kid,
                        "par_loss":model_output.loss_par,
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
            outputs["prediction"] = model_output.prediction_kid

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=outputs
        )

    def _evaluate(self, model_output, kid_labels):
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=model_output.loss,
            eval_metric_ops={
                "accuracy": tf.metrics.accuracy(kid_labels, model_output.prediction_kid),
                "mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(kid_labels,
                                                                              model_output.prediction_kid,
                                                                              self._model_configs.classes)
            }
        )

    def _build_model(self, features, kid_labels, par_labels, mode):
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

        if self._model_configs.pooling == 'concat':
            word_pool = _ConcatPooling()
        elif self._model_configs.pooling == 'ave':
            word_pool = _AveragePooling()
        else:
            print('Not implemented pooling')

        model = _Model(
            word_encoder=word_encoder,
            pooling=word_pool,
            dropout=self._model_configs.dropout,
            layers=[int(n) for n in self._model_configs.hidden_layers.split(",")],
            training=training,
            scope_name="Classification",
            kid_count=self._model_configs.kid_count
        )

        model_output = model(
            query = query,
            mask = mask,
            kid_labels = kid_labels,
            par_labels = par_labels
        )
        model_output.oneid = oneid
        return model_output

    def model_fn(self, features, labels, mode):
        if labels is not None:
            kid_labels = labels['kid_labels']
            par_labels = labels['par_labels']
        else:
            kid_labels = None
            par_labels = None
        model_output = self._build_model(features, kid_labels, par_labels, mode)

        if mode is tf.estimator.ModeKeys.TRAIN:
            return self._train(model_output, kid_labels, par_labels)
        elif mode is tf.estimator.ModeKeys.PREDICT:
            return self._predict(model_output)
        elif mode is tf.estimator.ModeKeys.EVAL:
            return self._evaluate(model_output, kid_labels)
