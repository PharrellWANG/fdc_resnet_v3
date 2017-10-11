terminal commands for freezing fdc resnet graph for c++
=======================================================

Export inference graph which only contain the architecture
----------------------------------------------------------

.. code-block:: bash

    $ python export_inference_graph.py


Freeze fdc resnet graph
-----------------------

Example from tensorflow slim lib:

.. code-block:: bash

    $ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 --config=cuda -k tensorflow/python/tools:freeze_graph

    $ bazel-bin/tensorflow/python/tools/freeze_graph \
        --input_graph=/tmp/inception_v3_inf_graph.pb \
        --input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
        --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
        --output_node_names=InceptionV3/Predictions/Reshape_1


.. code-block:: bash

    $ bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=/Users/Pharrell_WANG/workspace/models/resnet/graphs/resnet_inf_graph_for_fdc_32640.pb --input_checkpoint=/Users/Pharrell_WANG/workspace/models/resnet/log/model.ckpt-133049 --input_binary=true --output_graph=/Users/Pharrell_WANG/workspace/models/resnet/graphs/frozen_resnet_for_fdc_blk8x8_batchsize16320_step133049.pb --output_node_names=logits/fdc_output_node


Run it in c++
-------------

.. code-block:: bash

    $ bazel build tensorflow/examples/label_image:label_image

    $ bazel-bin/tensorflow/examples/label_image/label_image \
        --image=${HOME}/Pictures/flowers.jpg \
        --input_layer=init/fdc_input_node/Conv2D \
        --output_layer=logits/fdc_output_node \
        --graph=/Users/Pharrell_WANG/frozen_graphs/frozen_fdc_resnet_graph.pb \
        --labels=/Users/Pharrell_WANG/workspace/models/resnet/fdc_labels.txt \
        --input_mean=0 \
        --input_std=255




Quick Memo
----------

Fast intra angular Prediction (applicable to DMM1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

step 1: Build the binary
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ bazel build -c opt --config=cuda resnet/...

step 2: fdc training
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ bazel-bin/resnet/resnet_main --train_data_path='/Users/Pharrell_WANG/data/finalized/size_08/train_08x08.csv' --log_root='/Users/Pharrell_WANG/workspace/models/resnet/log' --train_dir='/Users/Pharrell_WANG/workspace/models/resnet/log/train' --dataset='fdc' --num_gpus=1


step 3: fdc evaluating
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ bazel-bin/resnet/resnet_main --eval_data_path='/Users/Pharrell_WANG/data/finalized/size_08/val_08x08.csv' --log_root="/Users/Pharrell_WANG/workspace/models/resnet/log" --eval_dir='/Users/Pharrell_WANG/workspace/models/resnet/log/eval' --mode=eval --dataset='fdc' --num_gpus=0


tensorboard
^^^^^^^^^^^

.. code-block:: bash

    $ tensorboard --logdir=/Users/Pharrell_WANG/workspace/models/resnet/log