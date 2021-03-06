# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FOR NEFESH COMPUTER
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from mcifar10 import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/mcifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/mcifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('retrain', False,
                            """Whether to retrain.""")
tf.app.flags.DEFINE_string('retrain_list', '',
                            """Names of layers to retraing.""")
tf.app.flags.DEFINE_boolean('print_params', False,
                            """Print values of parameters during training.""")

def train(retrain=False,retrain_list=None):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    if not retrain:
      train_op = cifar10.train(loss, global_step)
    else:
      train_op = cifar10.train(loss, global_step, retrain_list)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    ### RETRAINING START

    if FLAGS.retrain:
      if FLAGS.debug:
        print("GLOBAL =============================================================================")
        for v in tf.global_variables(): print(v.name)
        print("TRAINABLE =============================================================================")
        for v in tf.trainable_variables(): print(v.name)
        print("MOVING AVERAGES =============================================================================")
        for v in tf.moving_average_variables(): print(v.name)
      variables_to_restore = [v for v in tf.global_variables() if not v.name.split('/')[0] in retrain_list]
      variables_to_initialize = [v for v in tf.global_variables() if v.name.split('/')[0] in retrain_list]
      if FLAGS.debug:
        print("RESTORE =============================================================================")
        for v in variables_to_restore: print(v.name)
        print("INITIALIZE =============================================================================")
        for v in variables_to_initialize: print(v.name)
      saver_retrain = tf.train.Saver(variables_to_restore)
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if not (ckpt and ckpt.model_checkpoint_path):
        print('Yikes! No checkpoint file found at %s to retrain :-('%(FLAGS.checkpoint_dir))
        return
      # Build an initialization operation to run below.
      init = tf.variables_initializer(variables_to_initialize)
    else:
      # Build an initialization operation to run below.
      init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))

    if FLAGS.retrain:
      # Restores from checkpoint
      saver_retrain.restore(sess, ckpt.model_checkpoint_path)

    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    #print((tf.global_variables()[9].name,tf.global_variables()[9].eval(session=sess)[0][0]))
    #print((tf.global_variables()[10].name,tf.global_variables()[10].eval(session=sess)[0]))
    if FLAGS.print_params:
      print (tf.global_variables()[2].name)
      print (tf.global_variables()[2].eval(session=sess))
      print (tf.global_variables()[9].name)
      print (tf.global_variables()[9].eval(session=sess))
      print (tf.global_variables()[10].name)
      print (tf.global_variables()[10].eval(session=sess))
      print ("-------------------------------------------")

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
        #print((tf.global_variables()[9].name,tf.global_variables()[9].eval(session=sess)[0][0]))
        #print((tf.global_variables()[10].name,tf.global_variables()[10].eval(session=sess)[0]))
        if FLAGS.print_params:
          print (tf.global_variables()[2].name)
          print (tf.global_variables()[2].eval(session=sess))
          print (tf.global_variables()[9].name)
          print (tf.global_variables()[9].eval(session=sess))
          print (tf.global_variables()[10].name)
          print (tf.global_variables()[10].eval(session=sess))
          print ("-------------------------------------------")
      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if FLAGS.retrain:
    if FLAGS.retrain_list == '':
      FLAGS.retrain_list = ['softmax_linear']
    else:
      FLAGS.retrain_list = set(FLAGS.retrain_list.split(' ')+['softmax_linear'])
    print ("Will only retrain following layer(s): %s."%(' '.join(FLAGS.retrain_list)))
    if FLAGS.train_dir[-5:]=='train' and FLAGS.train_dir[-7:]!='train':
      FLAGS.train_dir=FLAGS.train_dir[0:-5]+'retrain'
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train(True,FLAGS.retrain_list)
  else:
    print ("Will train from scratch")
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
  tf.app.run()
