# added classification layer for emotion
# added classification loss for emotion
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope


# tf.app.flags.DEFINE_integer("num_units", 512, "Size of each model layer.")
def output_projection_layer(num_units, num_symbols, num_emotions, num_samples=None, name="output_projection"):
    def emotion_classification_fn(outputs):
        return layers.linear(outputs, num_emotions, scope=name)

    def output_fn(outputs):
        return layers.linear(outputs, num_symbols, scope=name)

    def sampled_sequence_loss(outputs, targets, masks):
        with variable_scope.variable_scope('decoder/%s' % name):
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])

            local_labels = tf.reshape(targets, [-1, 1])
            local_outputs = tf.reshape(outputs, [-1, num_units])
            local_masks = tf.reshape(masks, [-1])

            local_loss = tf.nn.sampled_softmax_loss(weights, bias, local_labels,
                                                    local_outputs, num_samples, num_symbols)
            local_loss = local_loss * local_masks

            loss = tf.reduce_sum(local_loss)
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12  # to avoid division by 0 for all-0 weights

            return loss / total_size

        def emotion_classification_loss(outputs, targets):
            with variable_scope.variable_scope('Emo/%s' % name):

    return output_fn, sampled_sequence_loss, emotion_classification_fn, emotion_classification_fn

