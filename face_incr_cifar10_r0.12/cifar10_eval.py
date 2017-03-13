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
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys

import numpy as np
import tensorflow as tf

from mcifar10 import cifar10, cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/mcifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/mcifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 30,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
def calculate_precision(model_predictions, correct_predictions):
	sum_avg = 0
	amount = 0
	num_predictions = 0
	for i in range(len(model_predictions)):
		#hits = 0
		#false_alarms = 0
		if(model_predictions[i] == 0):
			sum_avg += 0
		else:
			sum_avg += (correct_predictions[i]/model_predictions[i])
			#num_predictions += model_predictions[i]
			#hits += (correct_predictions[i])
			#false_alarms += (model_predictions[i]-correct_predictions[i])
			#print("Index: "+str(i)+" Hits: "+str(hits)+" False Alarm: "+str(false_alarms))
			amount += 1
	if(amount == 0):
		return 0
	return sum_avg/amount


def calculate_recall(correct_predictions, needed_predictions):
	sum_avg = 0
	amount = 0
	for i in range(len(correct_predictions)):
		hits = 0
		misses = 0
		if(needed_predictions[i] == 0):
			sum_avg += 0
		else:
			sum_avg += (correct_predictions[i]/needed_predictions[i])
			hits += correct_predictions[i]
			misses += (needed_predictions[i]-correct_predictions[i])
			print("Index: "+str(i)+" Hits: "+str(hits)+" Misses: "+str(misses)+" ")
			amount += 1
	if(amount == 0):
		return 0
	return sum_avg/amount

def calculate_false_positives(incorrect_predictions, needed_predictions):
	if (len(incorrect_predictions) == len(needed_predictions)):
		print("Yay! LENGHTS MATCH")
	sum_avg = 0
	amount = 0
	for class_index in range(len(incorrect_predictions)):
		false_positives = 0
		correct_rejects = 0
		negative_amt = 0
		for i in range(len(needed_predictions)):
			if(i != class_index):
				negative_amt += needed_predictions[i]
		if (incorrect_predictions[class_index] == 0):
			#print("Index: "+str(class_index)+" False Positives: 0 Correct Rejects: "+str(negative_amt)+" ")
			sum_avg += 0
			amount += 1
		else:
			#print(class_index)
			sum_avg += (incorrect_predictions[class_index]/negative_amt)
			false_positives += incorrect_predictions[class_index]
			correct_rejects += (negative_amt - incorrect_predictions[class_index])
			#print(incorrect_predictions[class_index])
			#print(negative_amt)
			#print("Index: "+str(class_index)+" False Positives: "+str(false_positives)+" Correct Rejects: "+str(correct_rejects))
			amount += 1
	if(amount == 0):
		return 0
	return sum_avg/amount


def eval_once(saver, summary_writer, top_k_op, summary_op, logits, labels, new_top_values_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      new_predicts_for_each_label = [0 for i in range(FLAGS.num_classes)]
      new_correct_predicts_for_each_label = [0 for i in range(FLAGS.num_classes)]
      new_incorrect_predicts_for_each_label = [0 for i in range(FLAGS.num_classes)]
      new_actual_amount_per_label = [0 for i in range(FLAGS.num_classes)]
      step = 0
      while step < num_iter and not coord.should_stop():
        [actual_labels, new_logits, predictions, new_values, new_indices] = sess.run([labels, logits, top_k_op, new_top_values_op[0], new_top_values_op[1]])
        #print(actual_labels)
        #print(predictions)
        #print("new indices: ")
        #print(new_indices)
        true_count += np.sum(predictions)
        for i in range(FLAGS.batch_size):
        	predict_class = new_indices[i][0]
        	new_predicts_for_each_label[predict_class] += 1
        	actual_label = actual_labels[i]
        	new_actual_amount_per_label[actual_label] += 1
	        if(predictions[i]):  #THIS SHOULD ONLY OCCUR WHEN "Class with top value" EQUALS "Actual Class Label"
	        	new_correct_predicts_for_each_label[predict_class] += 1
	        else:
	        	new_incorrect_predicts_for_each_label[predict_class] += 1
        step += 1
        print(step)
        print(coord.should_stop())
      #print(new_predicts_for_each_label)
      #print(new_correct_predicts_for_each_label)
      print("Megha's Precision: ")
      print(calculate_precision(new_predicts_for_each_label, new_correct_predicts_for_each_label))
      print("Megha's Recall/ TRUE POSITIVE Rate: ")
      print(calculate_recall(new_correct_predicts_for_each_label, new_actual_amount_per_label))
      print("Megha's False Positive Rate: ")
      print(calculate_false_positives(new_incorrect_predicts_for_each_label, new_actual_amount_per_label))

      # Compute precision @ 1.
      #print(logits[0,:].eval())
      #print(labels[1].eval())
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    new_top_values_op = tf.nn.top_k(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, logits, labels, new_top_values_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
