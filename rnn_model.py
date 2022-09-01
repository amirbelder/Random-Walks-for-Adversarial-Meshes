from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict
import copy

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import utils

from tensorflow import keras
layers = tf.keras.layers


class RnnWalkBase(tf.keras.Model):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn=None,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    super(RnnWalkBase, self).__init__(name='')

    self._classes = classes
    self._params = params
    self._model_must_be_load = model_must_be_load

    self._pooling_betwin_grus = 'pooling' in self._params.aditional_network_params
    self._bidirectional_rnn = 'bidirectional_rnn' in self._params.aditional_network_params

    self._init_layers()
    inputs = tf.keras.layers.Input(shape=(100, net_input_dim))
    self.build(input_shape=(1, 100, net_input_dim))
    outputs = self.call(inputs)
    if dump_model_visualization:
      tmp_model = keras.Model(inputs=inputs, outputs=outputs, name='WalkModel')
      tmp_model.summary(print_fn=self._print_fn)
      tf.keras.utils.plot_model(tmp_model, params.logdir + '/RnnWalkModel.png', show_shapes=True)

    self.manager = None
    if optimizer:
      if model_fn:
        #self.checkpoint = tf.train.Checkpoint(optimizer=copy.deepcopy(optimizer), model=self)
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
      else:
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
      self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self._params.logdir, max_to_keep=5)
      if model_fn: # Transfer learning
        self.load_weights(model_fn)
        self.checkpoint.optimizer = optimizer
      else:
        self.load_weights()
    else:
      self.checkpoint = tf.train.Checkpoint(model=self)
      if model_fn:
        self.load_weights(model_fn)
      else:
        self.load_weights(tf.train.latest_checkpoint(self._params.logdir))

  def _print_fn(self, st):
    with open(self._params.logdir + '/log.txt', 'at') as f:
      f.write(st + '\n')

  def load_weights(self, filepath=None):
    if filepath is not None and filepath.endswith('.keras'):
      super(RnnWalkBase, self).load_weights(filepath)
    elif filepath is None:
      status = self.checkpoint.restore(self.manager.latest_checkpoint)
      print(utils.color.BLUE, 'Starting from iteration: ', self.checkpoint.optimizer.iterations.numpy(), utils.color.END)
    else:
      filepath = filepath.replace('//', '/')
      status = self.checkpoint.restore(filepath)

  def save_weights(self, folder, step=None, keep=False):
    if self.manager is not None:
      self.manager.save()
    if keep:
      super(RnnWalkBase, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')
      #self.checkpoint.write(folder + '/learned_model2keep--' + str(step))



class RnnWalkNet(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
    else:
      self._layer_sizes = params.layer_sizes
    super(RnnWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = layers.BatchNormalization(axis=2)
      self._norm2 = layers.BatchNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    #rnn_layer = layers.LSTM
    rnn_layer = layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,      --->> very slow!! (tf2.1)
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru1 = layers.Bidirectional(self._gru1)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru2 = layers.Bidirectional(self._gru2)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           #trainable=False,
                           #activation='sigmoid',
                           dropout=self._params.net_gru_dropout,
                           #recurrent_dropout=self._params.net_gru_dropout,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru3 = layers.Bidirectional(self._gru3)
      print('Using Bidirectional GRUs.')
    self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    self._norm_input = False
    if self._norm_input:
      self._norm_features = layers.LayerNormalization(axis=-1, trainable=False)

  # @tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True, mask=None):
    if self._norm_input:
      model_ftrs = self._norm_features(model_ftrs)
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x1 = self._gru1(x, training=training)
    if self._pooling_betwin_grus:
      x1 = self._pooling(x1)
      if mask is not None:
        mask = mask[:, ::2]
    x2 = self._gru2(x1, training=training)
    if self._pooling_betwin_grus:
      x2 = self._pooling(x2)
      if mask is not None:
        mask = mask[:, ::2]
    x3 = self._gru3(x2, training=training, mask=mask)
    x = x3

    #if self._params.one_label_per_model:
    #  x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x

  def call_dbg(self, model_ftrs, classify=True, skip_1st=True, training=True, get_layer=None):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    if get_layer == 'input':
      return x
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc1':
      return x
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc2':
      return x
    x = self._gru1(x, training=training)
    if get_layer == 'gru1':
      return x
    x = self._gru2(x, training=training)
    if get_layer == 'gru2':
      return x
    x = self._gru3(x, training=training)
    if get_layer == 'gru3':
      return x

    if self._params.one_label_per_model:
      x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x


class RnnManifoldWalkNet(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
    else:
      self._layer_sizes = params.layer_sizes
    super(RnnManifoldWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = layers.BatchNormalization(axis=2)
      self._norm2 = layers.BatchNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    #rnn_layer = layers.LSTM
    rnn_layer = layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,      --->> very slow!! (tf2.1)
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru1 = layers.Bidirectional(self._gru1)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru2 = layers.Bidirectional(self._gru2)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           #trainable=False,
                           #activation='sigmoid',
                           dropout=self._params.net_gru_dropout,
                           #recurrent_dropout=self._params.net_gru_dropout,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru3 = layers.Bidirectional(self._gru3)
      print('Using Bidirectional GRUs.')
    self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    self._norm_input = False
    if self._norm_input:
      self._norm_features = layers.LayerNormalization(axis=-1, trainable=False)

  # @tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True, mask=None):
    if self._norm_input:
      model_ftrs = self._norm_features(model_ftrs)
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x1 = self._gru1(x, training=training)
    if self._pooling_betwin_grus:
      x1 = self._pooling(x1)
      if mask is not None:
        mask = mask[:, ::2]
    x2 = self._gru2(x1, training=training)
    if self._pooling_betwin_grus:
      x2 = self._pooling(x2)
      if mask is not None:
        mask = mask[:, ::2]
    x3 = self._gru3(x2, training=training, mask=mask)
    x = x3

    if classify:
      x = self._fc_last(x)
    return x

  def call_dbg(self, model_ftrs, classify=True, skip_1st=True, training=True, get_layer=None):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    if get_layer == 'input':
      return x
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc1':
      return x
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc2':
      return x
    x = self._gru1(x, training=training)
    if get_layer == 'gru1':
      return x
    x = self._gru2(x, training=training)
    if get_layer == 'gru2':
      return x
    x = self._gru3(x, training=training)
    if get_layer == 'gru3':
      return x

    if self._params.one_label_per_model:
      x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x


class Unsupervised_RnnWalkNet(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
    else:
      self._layer_sizes = params.layer_sizes
    if params.network_task == 'features_extraction':
      self.features_extraction = True
    else:
      self.features_extraction = False

    super(Unsupervised_RnnWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = layers.BatchNormalization(axis=2)
      self._norm2 = layers.BatchNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    #rnn_layer = layers.LSTM
    rnn_layer = layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,      --->> very slow!! (tf2.1)
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru1 = layers.Bidirectional(self._gru1)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru2 = layers.Bidirectional(self._gru2)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           #trainable=False,
                           #activation='sigmoid',
                           dropout=self._params.net_gru_dropout,
                           #recurrent_dropout=self._params.net_gru_dropout,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    if self._bidirectional_rnn:
      self._gru3 = layers.Bidirectional(self._gru3)
      print('Using Bidirectional GRUs.')

    if not self.features_extraction:
      self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                   kernel_initializer=initializer)
    self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    self._l2_normalization = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    self._norm_input = False
    if self._norm_input:
      self._norm_features = layers.LayerNormalization(axis=-1, trainable=False)

  # @tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True, mask=None):
    if self._norm_input:
      model_ftrs = self._norm_features(model_ftrs)
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x1 = self._gru1(x, training=training)
    if self._pooling_betwin_grus:
      x1 = self._pooling(x1)
      if mask is not None:
        mask = mask[:, ::2]
    x2 = self._gru2(x1, training=training)
    if self._pooling_betwin_grus:
      x2 = self._pooling(x2)
      if mask is not None:
        mask = mask[:, ::2]
    x3 = self._gru3(x2, training=training, mask=mask)
    x = x3

    #if self._params.one_label_per_model:
    #  x = x[:, -1, :]
    if self.features_extraction:
      x = self._l2_normalization(x)
    elif classify:
      x = self._fc_last(x)
      # L2 normalization is meeded when working with triplet loss
      x = self._l2_normalization(x)

    return x

  def call_dbg(self, model_ftrs, classify=True, skip_1st=True, training=True, get_layer=None):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    if get_layer == 'input':
      return x
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc1':
      return x
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc2':
      return x
    x = self._gru1(x, training=training)
    if get_layer == 'gru1':
      return x
    x = self._gru2(x, training=training)
    if get_layer == 'gru2':
      return x
    x = self._gru3(x, training=training)
    if get_layer == 'gru3':
      return x

    if self._params.one_label_per_model:
      x = x[:, -1, :]

    if classify and not self.features_extraction:
      x = self._fc_last(x)
    return x

class AttentionWalkNet(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               layer_sizes={'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512, 'gru_dec1': 1024, 'gru_dec2': 512},
               model_must_be_load=False,
               optimizer=None):
    self._layer_sizes = layer_sizes
    super(AttentionWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = 1
    if self._use_norm_layer:
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._gru1 = layers.GRU(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #activation='sigmoid',
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru2 = layers.GRU(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #activation='sigmoid',
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)

    self._gru3 = layers.GRU(self._layer_sizes['gru3'], time_major=False, return_sequences=True, return_state=True,
                            #activation='sigmoid',
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                            bias_regularizer=kernel_regularizer)
    self._attention_layer = BahdanauAttention(10)

    self._gru_decode_1 = layers.GRU(self._layer_sizes['gru_dec1'], time_major=False, return_sequences=True, return_state=False,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru_decode_2 = layers.GRU(self._layer_sizes['gru_dec2'], time_major=False, return_sequences=True, return_state=False,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)

    self._fc_last = layers.Dense(self._classes, activation='sigmoid', kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)

  #@tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    model_ftrs_ = x

    # Encoder
    # -------
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x = self._gru1(x)
    x = self._gru2(x)
    output, hidden = self._gru3(x)

    # Attention
    # ---------
    context_vector, attention_weights = self._attention_layer(hidden, output)

    # Decoder
    # -------
    x = tf.concat([tf.expand_dims(context_vector, 1), model_ftrs_], axis=-1)
    x = self._gru_decode_1(x)
    x = self._gru_decode_2(x)
    x = self._fc_last(x)

    return x

class RnnStrideWalkNet(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               layer_sizes={'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 1024},
               model_must_be_load=False):
    self._layer_sizes = layer_sizes
    super(RnnStrideWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = 1
    if self._use_norm_layer:
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._gru1 = layers.GRU(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #activation='sigmoid',
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru2 = layers.GRU(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #activation='sigmoid',
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru3 = layers.GRU(self._layer_sizes['gru3'], time_major=False, return_sequences=True, return_state=False,
                            #activation='sigmoid',
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                            bias_regularizer=kernel_regularizer)
    self._fc_last = layers.Dense(self._classes, activation='sigmoid', kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')
    self._up_sampling = layers.UpSampling1D(size=2)

  #@tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x = self._gru1(x)
    before_pooling = x
    x = self._pooling(x)
    x = self._gru2(x)
    x = self._gru3(x)
    x = self._up_sampling(x)
    x = x[:, :before_pooling.shape[1], :] + before_pooling

    if self._params.one_label_per_model:
      x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x

def set_up_rnn_walk_model():
  _layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
  last_layer_activation = 'softmax'
  _classes = 40
  training = True
  one_label_per_model = True
  classify = True

  input = keras.Input(shape=(28, 28, 1), name='original_img')

  kernel_regularizer = tf.keras.regularizers.l2(0.0001)
  initializer = tf.initializers.Orthogonal(3)
  _norm1 = tfa.layers.InstanceNormalization(axis=2)
  _norm2 = tfa.layers.InstanceNormalization(axis=2)
  _fc1 = layers.Dense(_layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                           kernel_initializer=initializer)
  _fc2 = layers.Dense(_layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                           kernel_initializer=initializer)
  _gru1 = layers.GRU(_layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                          recurrent_initializer=initializer, kernel_initializer=initializer,
                          kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
  _gru2 = layers.GRU(_layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                          recurrent_initializer=initializer, kernel_initializer=initializer,
                          kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
  _gru3 = layers.GRU(_layer_sizes['gru3'], time_major=False, return_sequences=True, return_state=False,
                          recurrent_initializer=initializer, kernel_initializer=initializer,
                          kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                          bias_regularizer=kernel_regularizer)
  _fc_last = layers.Dense(_classes, activation=last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                               kernel_initializer=initializer)

  inputs = keras.Input(shape=(100, 4,))
  x = inputs
  x = _fc1(x)
  x = _norm1(x, training=training)
  x = tf.nn.relu(x)
  x = _fc2(x)
  x = _norm2(x, training=training)
  x = tf.nn.relu(x)
  x = _gru1(x)
  x = _gru2(x)
  x = _gru3(x)

  if one_label_per_model:
    x = x[:, -1, :]

  if classify:
    x = _fc_last(x)

  outputs = x

  model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')

  return model


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class GruAndVanillaLayer(tf.keras.Model): # TODO: to inherent from layer, not model
  def __init__(self, units_, initializer, regularizer):
    super(GruAndVanillaLayer, self).__init__(name='')
    units = int(units_ / 2)
    regularizer_ = tf.keras.regularizers.l2(100)
    self._gru = layers.GRU(units, time_major=False, return_sequences=True, return_state=False,
                    kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer,
                    kernel_initializer=initializer, recurrent_initializer=initializer)

    self._simple_rnn = layers.SimpleRNN(units, time_major=False, return_sequences=True, return_state=False, activation=None,
                    kernel_regularizer=regularizer_, recurrent_regularizer=regularizer_, bias_regularizer=regularizer_,
                    kernel_initializer=initializer, recurrent_initializer=initializer)

  def call(self, x):
    x1 = self._gru(x)
    x2 = self._simple_rnn(x)
    r = tf.concat((x1, x2), axis=2)
    return r


def show_model():
  def fn(to_print):
    print(to_print)
  if 1:
    params = EasyDict({'n_classes': 3, 'net_input_dim': 3, 'batch_size': 32, 'last_layer_activation': 'softmax',
                       'one_label_per_model': True, 'logdir': '.'})
    params.net_input_dim = 3 + 5
    model = RnnWalkNet(params, classes=3, net_input_dim=3, model_fn=None)
  else:
    model = set_up_rnn_walk_model()
    tf.keras.utils.plot_model(model, "RnnWalkModel.png", show_shapes=True)
    model.summary(print_fn=fn)

if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu(0)
  show_model()
