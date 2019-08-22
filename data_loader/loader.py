# -*- coding: utf-8 -*-#
import tensorflow as tf



class OdpsDataLoader:
    def __init__(self, table_name, max_length, mode, repeat=None, batch_size=128, shuffle=2000, slice_id=0, slice_count=1):
        # Avoid destroying input parameter
        self._table_name = table_name
        self._max_length = max_length
        self._slice_id = slice_id
        self._slice_count = slice_count
        self._batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self._mode = mode

    def _text_content_parser(self, text, max_length, shift_no=True):
        word_strs = tf.string_split([text], " ")
        return tf.string_to_number(word_strs.values, out_type=tf.int64)[:max_length]+ (1 if shift_no else 0), \
               tf.ones([tf.minimum(tf.shape(word_strs)[-1], max_length)])

    def _train_data_parser(self, oneid, content, kid_label, par_label):
        words, masks = self._text_content_parser(content, self._max_length)
        return {
            "oneid": oneid,
            "words": words,
            "masks": masks
        }, {"kid_labels":kid_label, "par_labels":par_label}

    def _test_data_parser(self, oneid, content):
        words, masks = self._text_content_parser(content, self._max_length)
        return {
                   "oneid": oneid,
                   "words": words,
                   "masks": masks
               }, tf.constant(0, dtype=tf.int32) # fake label

    def _train_data_fn(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TableRecordDataset(
                self._table_name,
                record_defaults=["", "", 0, 0],
                slice_id=self._slice_id,
                slice_count=self._slice_count
            )

            dataset = dataset.map(self._train_data_parser, num_parallel_calls=4)
            if self._shuffle > 0:
                dataset = dataset.shuffle(self._shuffle)

            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            dataset = dataset.prefetch(40000)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "oneid": [],
                        "words": [self._max_length],
                        "masks": [self._max_length],
                    },
                    {
                        "kid_labels":[],
                        "par_labels":[]
                    })
            )

            return dataset.make_one_shot_iterator().get_next()

    def _test_data_fn(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TableRecordDataset(
                self._table_name,
                record_defaults=["", ""],
                slice_id=self._slice_id,
                slice_count=self._slice_count
            )

            dataset = dataset.map(self._test_data_parser, num_parallel_calls=4)
            if self._shuffle > 0:
                dataset = dataset.shuffle(self._shuffle)

            if self._repeat != 1:
                dataset = dataset.repeat(self._repeat)

            dataset = dataset.prefetch(40000)
            dataset = dataset.padded_batch(
                self._batch_size,
                padded_shapes=(
                    {
                        "oneid": [],
                        "words": [self._max_length],
                        "masks": [self._max_length]
                    }, [])
            )

            return dataset.make_one_shot_iterator().get_next()

    def input_fn(self):
        return self._train_data_fn() if self._mode else self._test_data_fn()

