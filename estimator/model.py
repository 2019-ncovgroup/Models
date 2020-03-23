import tensorflow as tf

DEFAULT_OPTIMIZER = 'sgd_momentum'
ALLOWED_OPTIMIZERS = ['sgd', 'sgd_momentum', 'adam']
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
LOSS_SCALE = 1  # No loss scaling by default


def build_network(features, params):

    x = tf.layers.dense(features, 250, activation=tf.nn.relu)
    x = tf.layers.dropout(x, params.get('dropout_rate', 0.1))

    x = tf.layers.dense(x, 125, activation=tf.nn.relu)
    x = tf.layers.dropout(x, params.get('dropout_rate', 0.1))

    x = tf.layers.dense(x, 60, activation=tf.nn.relu,)
    x = tf.layers.dropout(x, params.get('dropout_rate', 0.1))

    x = tf.layers.dense(x, 30, activation=tf.nn.relu)
    x = tf.layers.dropout(x, params.get('dropout_rate', 0.1))

    predictions = tf.layers.dense(x, 1, activation=tf.nn.relu)

    return predictions


def model_fn(features, labels, mode, params=None):
    loss = None
    train_op = None
    training_hooks = None
    eval_metric_ops = None

    optimizer = params.get('optimizer', DEFAULT_OPTIMIZER)
    if optimizer not in ALLOWED_OPTIMIZERS:
        raise ValueError('optimizer must be one of: {}'.format(
            ALLOWED_OPTIMIZERS))

    dtype = tf.float16 if params.get("fp16") else DEFAULT_DTYPE
    if dtype not in ALLOWED_TYPES:
        raise ValueError('dtype must be one of: {}'.format(
            ALLOWED_TYPES))

    features = tf.cast(features, dtype=dtype)
    if labels is not None:
        labels = tf.cast(labels, dtype=dtype)

    mixed_precision = params.get('mixed_precision', dtype != DEFAULT_DTYPE)
    loss_scale = float(params.get('loss_scale', LOSS_SCALE))
    custom_getter = _custom_dtype_getter if mixed_precision else None

    params["training"] = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('uno', dtype=dtype, custom_getter=custom_getter):
        predictions = build_network(features, params)

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=predictions)
        loss = tf.cast(loss, dtype=dtype)

        if params.get("log_metrics"):
            # Add metrics to TensorBoard summaries
            metrics = uno_metrics(labels, predictions)
            tf.summary.scalar('mse', metrics['mse'][1])
            tf.summary.scalar('mae', metrics['mae'][1])
            tf.summary.scalar('r2', metrics['r2'][1])

            logging_hook = tf.train.LoggingTensorHook(
                tensors={
                    "loss": loss,
                    "mse": metrics['mse'][0],
                    "mae": metrics['mae'][0],
                    "r2": metrics['r2'][0],
                    "step": tf.train.get_global_step()},
                every_n_iter=params.get('train_steps'),
                at_end=True)
            training_hooks = [logging_hook]
            eval_metric_ops = metrics

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        if 'learning_rate' in params:
            learning_rate = params['learning_rate']
        elif params.get('xla_compile'):
            # Search for 'xla_compile' in ml_uno.py for more
            # information about this special case.
            learning_rate = params['lr_schedule'][0][0]
        else:
            from cerebras.tf.models.uno.utils import tf_learning_rate
            learning_rate = tf_learning_rate(global_step,
                                             params['lr_schedule'])
        # Choose the right optimizer
        if optimizer == 'sgd_momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.9,
                name='sgd_momentum')
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                name='adam'
            )
        elif optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate,
                name='sgd')

        # Apply loss-scaling if training in mixed precision
        if mixed_precision and loss_scale > 1:
            scaled_grads_vars = optimizer.compute_gradients(
                loss * loss_scale)
            unscaled_grads_vars = [
                (g / loss_scale, v) for g, v in scaled_grads_vars]
            train_op = optimizer.apply_gradients(
                unscaled_grads_vars, tf.train.get_global_step())
        else:
            scaled_grads_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(
                scaled_grads_vars, tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        training_hooks=training_hooks)


def uno_metrics(labels, predictions):
    """
    Get MSE, MAE, and R2 metrics for Uno.
    Arguments:
        labels (tf.Tensor)
            Tensor containing batch of labels
        predictions: (tf.Tensor)
            Tensor containing model predictions
    Returns:
        (dict of (tf.Tensor, update_op))):
            Dictionary of metrics and corresponding
            update operations.
    """
    mse_op = tf.metrics.mean_squared_error(
        labels=labels, predictions=predictions)
    mae_op = tf.metrics.mean_absolute_error(
        labels=labels, predictions=predictions)
    r2_op = tf.metrics.mean(r_squared(labels, predictions))
    return {'mse': mse_op, 'mae': mae_op, 'r2': r2_op}


def r_squared(labels, predictions):
    """
    Computes R2 of the model.
    Arguments:
        labels (tf.Tensor)
            Tensor containing batch of labels
        predictions: (tf.Tensor)
            Tensor containing model predictions
    Returns:
        (tf.Tensor):
            Scalar Tensor representing the Rsquared
    """
    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    r_sq = 1.0 - tf.math.divide(unexplained_error, total_error)

    return r_sq


def _custom_dtype_getter(getter, name, shape, dtype, *args, **kwargs):
    """
    Custom variable getter that forces trainable variables to be stored in
    fp32 precision and then casts them to the training precision (e.g. fp16),
    when training with mixed precision.
    Args:
        getter:
            The underlying variable getter, that has the same signature as
            tf.get_variable and returns a variable.
        name:
            The name of the variable to get.
        shape:
            The shape of the variable to get.
        dtype:
            The dtype of the variable to get. Note that if this is a low
            precision dtype, the variable will be created as a tf.float32
            variable, then cast to the appropriate dtype
        *args:
            Additional arguments to pass unmodified to getter.
        **kwargs:
            Additional keyword arguments to pass unmodified to getter.
    Returns:
        A variable which is cast to fp16 if necessary.
    """
    if dtype in CASTABLE_TYPES:
        var = getter(name, shape, tf.float32, *args, **kwargs)
        return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
        return getter(name, shape, dtype, *args, **kwargs)

