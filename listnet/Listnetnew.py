from absl import flags

import numpy as np
import six
import tensorflow as tf
import tensorflow_ranking as tfr


class ExpandBatchLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExpandBatchLayer, self).__init__(**kwargs)

    def call(self, input):
        queries, docs = input
        batch, num_docs, embedding_dims = tf.unstack(tf.shape(docs))
        expanded_queries = tf.gather(queries, tf.zeros([num_docs], tf.int32), axis=1)
        return tf.concat([expanded_queries, docs], axis=-1)


class Listnet:
    def __init__(self):
        query_input = tf.keras.layers.Input(shape=(1, 10,), dtype=tf.float32, name='query')
        docs_input = tf.keras.layers.Input( dtype=tf.float32, name='docs')

        expand_batch = ExpandBatchLayer(name='expand_batch')
        dense_1 = tf.keras.layers.Dense(units=3, activation='relu', name='dense_1')
        dense_out = tf.keras.layers.Dense(units=1, activation='relu', name='scores')
        scores_prob_dist = tf.keras.layers.Dense(units=100, activation='softmax', name='scores_prob_dist')

        expanded_batch = expand_batch([query_input, docs_input])
        dense_1_out = dense_1(expanded_batch)
        scores = tf.keras.layers.Flatten()(dense_out(dense_1_out))
        model_out = scores_prob_dist(scores)

        self.model = tf.keras.models.Model(inputs=[query_input, docs_input], outputs=[model_out])

        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9),
                      loss=tf.keras.losses.KLDivergence())

    def fit(self, x_train, y_train, qids):
        raw_relevance_grades = tf.constant(y_train, dtype=tf.float32)
        relevance_grades_prob_dist = tf.nn.softmax(raw_relevance_grades)

        hist = self.model.fit(
            [qids, x_train],
            relevance_grades_prob_dist,
            epochs=50,
            verbose=False
        )

    def predict(self, x_test, qids):
        self.model.predict([qids, x_test])
