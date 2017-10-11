"""CIFAR dataset input module.
"""

import tensorflow as tf


def build_input(dataset, data_path, batch_size, mode, block_size, num_of_classes):
    """Build CIFAR image and labels.

    Args:
      dataset: can be 'cifar10' or 'cifar100' or 'fdc'.
      data_path: Filename for data.
      batch_size: Input batch size.
      mode: Either 'train' or 'eval'.
      block_size: this is only for fdc. can be 8, 16, 32 or 64, default 32
      num_of_classes: ..
    Returns:
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
    Raises:
      ValueError: when the specified dataset is not supported.
    """
    if dataset == 'fdc':
        # blk_size default is 32
        # for convention, still use image_size as name
        image_size = block_size
        num_classes = num_of_classes
        depth = 1

        filename_queue = tf.train.string_input_producer([data_path])

        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        # Default values, in case of empty columns. Also specifies the type of the
        # decoded result.

        record_defaults = [[1] for _ in range(block_size * block_size + 1)]

        # list_of_256plus1_columns ==> one_line_in_csv
        one_line_in_csv = tf.decode_csv(value,
                                        record_defaults=record_defaults)

        image = tf.stack(
            one_line_in_csv[0:len(one_line_in_csv) - 1])
        label = tf.stack(
            one_line_in_csv[len(one_line_in_csv) - 1])

        depth_major = tf.reshape(image,
                                 [depth, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        label = tf.cast(label, tf.int32)
        label = tf.reshape(label, (1,))
        # print('===================')
        # print(image.shape)
        # print(label.shape)
        # image = np.multiply(image, 1.0 / 255.0)

        if mode == 'train':
            example_queue = tf.RandomShuffleQueue(
                capacity=16 * batch_size,
                min_after_dequeue=8 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[image_size, image_size, depth], [1]])
            num_threads = 16
        else:
            example_queue = tf.FIFOQueue(
                3 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[image_size, image_size, depth], [1]])
            num_threads = 1

        example_enqueue_op = example_queue.enqueue([image, label])
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
            example_queue, [example_enqueue_op] * num_threads))

        # Read 'batch' labels + image from the example queue.
        images, labels = example_queue.dequeue_many(batch_size)
        labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
        labels = tf.sparse_to_dense(
            tf.concat(values=[indices, labels], axis=1),
            [batch_size, num_classes], 1.0, 0.0)

        assert len(images.get_shape()) == 4
        assert images.get_shape()[0] == batch_size
        assert images.get_shape()[-1] == 1
        assert len(labels.get_shape()) == 2
        assert labels.get_shape()[0] == batch_size
        assert labels.get_shape()[1] == num_classes

        # with tf.Session() as sess:
        #     # Start populating the filename queue.
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(coord=coord)
        #
        #     for i in range(1):
        #         # Retrieve a single instance:
        #         image, label = sess.run([image, label])
        #         print(image)
        #         print(image.shape)
        #         print(label)
        #
        #     coord.request_stop()
        #     coord.join(threads)

        # Display the training image in the visualizer.
        tf.summary.image('images', images, 10)
        return images, labels

    else:
        image_size = 32
        if dataset == 'cifar10':
            label_bytes = 1
            label_offset = 0
            num_classes = 10
        elif dataset == 'cifar100':
            label_bytes = 1
            label_offset = 1
            num_classes = 100
        else:
            raise ValueError('Not supported dataset %s', dataset)

        depth = 3
        image_bytes = image_size * image_size * depth
        record_bytes = label_bytes + label_offset + image_bytes

        data_files = tf.gfile.Glob(data_path)
        file_queue = tf.train.string_input_producer(data_files, shuffle=True)
        # Read examples from files in the filename queue.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(file_queue)

        # Convert these examples to dense labels and processed images.
        record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
        label = tf.cast(tf.slice(record, [label_offset], [label_bytes]),
                        tf.int32)
        # Convert from string to [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]),
                                 [depth, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        if mode == 'train':
            image = tf.image.resize_image_with_crop_or_pad(
                image, image_size + 4, image_size + 4)
            image = tf.random_crop(image, [image_size, image_size, 3])
            image = tf.image.random_flip_left_right(image)
            # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
            # image = tf.image.random_brightness(image, max_delta=63. / 255.)
            # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            image = tf.image.per_image_standardization(image)

            example_queue = tf.RandomShuffleQueue(
                capacity=16 * batch_size,
                min_after_dequeue=8 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[image_size, image_size, depth], [1]])
            num_threads = 16
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, image_size, image_size)
            image = tf.image.per_image_standardization(image)

            example_queue = tf.FIFOQueue(
                3 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[image_size, image_size, depth], [1]])
            num_threads = 1

        example_enqueue_op = example_queue.enqueue([image, label])
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
            example_queue, [example_enqueue_op] * num_threads))

        # Read 'batch' labels + images from the example queue.
        images, labels = example_queue.dequeue_many(batch_size)
        labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
        labels = tf.sparse_to_dense(
            tf.concat(values=[indices, labels], axis=1),
            [batch_size, num_classes], 1.0, 0.0)

        assert len(images.get_shape()) == 4
        assert images.get_shape()[0] == batch_size
        assert images.get_shape()[-1] == 3
        assert len(labels.get_shape()) == 2
        assert labels.get_shape()[0] == batch_size
        assert labels.get_shape()[1] == num_classes

        # Display the training images in the visualizer.
        tf.summary.image('images', images)
        return images, labels
