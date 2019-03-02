import tensorflow as tf

def _lstm_cell(n_hidden, n_layers, name=None):
    """select proper lstm cell."""
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True, name=name or 'basic_lstm_cell')
    if n_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(
             n_hidden, state_is_tuple=True, reuse=reuse) for _ in range(n_layers)])
    return cell

# def _lstm_cell(n_hidden, n_layers, reuse=False, keep_prob=1.):
#   """select proper lstm cell."""
#   cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=keep_prob, reuse=reuse)
#   if n_layers > 1:
#       cell = tf.contrib.rnn.MultiRNNCell(
#           [tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden,
#            dropout_keep_prob=keep_prob, reuse=reuse) for _ in range(n_layers)])
#   return cell

def create_inite_state(n_hidden, n_layers, batch_size, scope=None):

  with tf.variable_scope(scope or 'init_state'):
    lstm_tuple = tf.contrib.rnn.LSTMStateTuple

    if n_layers > 1:
      new_state = [lstm_tuple(tf.tile(tf.get_variable('h{0}'.format(i), [1, n_hidden]), [batch_size, 1]), 
                              tf.tile(tf.get_variable('c{0}'.format(i), [1, n_hidden]), [batch_size, 1])) for i in xrange(n_layers)]
      init_state = tuple(new_state)
    else:
      init_state = lstm_tuple(tf.tile(tf.get_variable('h0', [1, n_hidden]), [batch_size, 1]),
                              tf.tile(tf.get_variable('c0', [1, n_hidden]), [batch_size, 1]))

  return init_state


def mlp(inputs, mlp_hidden=256, mlp_layers=2, scope=None, keep_prob=None):
  """build an MLP."""
  with tf.variable_scope(scope or 'mlp'):
    outputs = inputs
    for i in xrange(mlp_layers):
      if keep_prob is not None:
        outputs = tf.tanh(
            linear_layer(
                tf.nn.dropout(outputs, keep_prob), mlp_hidden, True, scope=('l' + str(i))))
      else:
        outputs = tf.tanh(
            linear_layer(
                outputs, mlp_hidden, True, scope=('l' + str(i))))
  return outputs


def linear_layer(inputs,
                 output_size,
                 bias=True,
                 bias_start_zero=False,
                 matrix_start_zero=False,
                 scope=None):
  """Define a linear connection that can customise the parameters."""

  shape = inputs.get_shape().as_list()

  if len(shape) != 2:
    raise ValueError('Linear is expecting 2D arguments: %s' % str(shape))
  if not shape[1]:
    raise ValueError('Linear expects shape[1] of arguments: %s' % str(shape))
  input_size = shape[1]

  # Now the computation.
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix = tf.get_variable(
          'Matrix', [input_size, output_size],
          initializer=tf.constant_initializer(0))
    else:
      matrix = tf.get_variable('Matrix', [input_size, output_size])
    res = tf.matmul(inputs, matrix)
    if not bias:
      return res
    if bias_start_zero:
      bias_term = tf.get_variable(
          'Bias', [output_size], initializer=tf.constant_initializer(0))
    else:
      bias_term = tf.get_variable('Bias', [output_size])
    output = res + bias_term
  return output

def Conv2D(inputs,
           num_outputs,
           kernel_size,
           strides,
           scope=None,
           activation=tf.nn.leaky_relu,
           trainable=True,
           reuse=False):
    outputs = tf.contrib.layers.conv2d(inputs=inputs,
                                       num_outputs=num_outputs,
                                       kernel_size=kernel_size,
                                       stride=strides,
                                       padding='SAME',
                                       activation_fn=activation,
                                       trainable=trainable,
                                       reuse=reuse,
                                       scope=scope or 'conv2d')
    return outputs

def Conv1D(inputs,
           num_outputs,
           kernel_size,
           strides,
           scope=None,
           activation=tf.nn.leaky_relu,
           trainable=True,
           reuse=False):
    outputs = tf.contrib.layers.conv1d(inputs=inputs,
                                       num_outputs=num_outputs,
                                       kernel_size=kernel_size,
                                       stride=strides,
                                       padding='SAME',
                                       activation_fn=activation,
                                       trainable=trainable,
                                       reuse=reuse,
                                       scope=scope or 'conv2d')
    return outputs

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)