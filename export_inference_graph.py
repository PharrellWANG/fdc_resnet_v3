# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
import resnet_model


tf.app.flags.DEFINE_string(
	'model_name', 'resnet', 'The name of the architecture to save.')

tf.app.flags.DEFINE_boolean(
	'is_training', False,
	'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
	'image_size', 8,
	'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
	'number_of_classes', 28,
	'number of classes')

tf.app.flags.DEFINE_integer(
	'batch_size', 1,
	'Batch size for the exported model. Defaulted to ``int(1)``')

tf.app.flags.DEFINE_string('dataset', 'fdc',
													 'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_string(
	'output_file',
	'/Users/Pharrell_WANG/workspace/models/resnet/graphs/resnet_inf_graph_for_fdc.pb',
	'Where to save the resulting file to.')

FLAGS = tf.app.flags.FLAGS


def main(_):
	if not FLAGS.output_file:
		raise ValueError('You must supply the path to save to with --output_file')
	tf.logging.set_verbosity(tf.logging.INFO)
	with tf.Graph().as_default() as graph:
		image_size = FLAGS.image_size
		placeholder = tf.placeholder(name='input', dtype=tf.float32,
																 shape=[FLAGS.batch_size,
																				image_size,
																				image_size,
																				1])
		hps = resnet_model.HParams(dataset_name=FLAGS.dataset,
															 batch_size=FLAGS.batch_size,
															 num_classes=FLAGS.number_of_classes,
															 min_lrn_rate=0.0001,
															 lrn_rate=0.1,
															 num_residual_units=5,
															 use_bottleneck=False,
															 weight_decay_rate=0.0002,
															 relu_leakiness=0.1,
															 optimizer='mom')
		# the third param  : ``label``, here we set label to None for eval mode.
		# the fourth param : ``mode``, it must be ``eval`` for exporting graph
		model = resnet_model.ResNet(hps, placeholder, None, 'eval')
		model.build_graph()
		graph_def = graph.as_graph_def()
		with gfile.GFile(FLAGS.output_file, 'wb') as f:
			f.write(graph_def.SerializeToString())


if __name__ == '__main__':
	tf.app.run()
