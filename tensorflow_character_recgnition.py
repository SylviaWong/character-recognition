"""

    Author: Huang Qianying
    Date: 2018.8

"""

import string, os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
import tensorflow as tf
import datetime as dt

# set the folder path
dir_name = 'data'
label_name = 'label_list.txt'
file_name = 'file_list.txt'

# Parameters
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
NUM_CHANNELS = 1
BATCH_SIZES = 100
CAPACITY = 2858  # 2,858 items
MIN_DEQUEUE = 100
CLASSES_NUM = 4

learning_rate = 0.01
batch_size = BATCH_SIZES
batch_num = int(CAPACITY / batch_size)
train_epoch = 10

# Network Parameters
n_input = 4096  # data input (img shape: 64*64)
n_classes = 4  # total classes
dropout = 0.8  # Dropout, probability to keep units

# set the file path
# files = os.listdir(dir_name)
files = open(file_name, "r")
paths = []
for f in files:
    paths.append(dir_name + os.sep + f.strip('\n'))
    # print(dir_name + os.sep + f)

f = open(label_name, "r")
labelss = []
for line in f:
    labels = [0] * CLASSES_NUM
    labels[int(line)] = 1
    labelss.append(labels)

# get the label
label_y = np.zeros((CAPACITY, 4))
# label_y[:, 0] = labels
# label_y[:, 1] = 1-labels

T_ind = random.sample(range(0, CAPACITY), CAPACITY)
# print(T_ind)
# print(len(T_ind))

random_paths = []
random_labels = []
for i in range(0, len(T_ind)):
    random_paths.append(paths[int(T_ind[i])])
    random_labels.append(labelss[int(T_ind[i])])


def get_batch(image, label, image_h, image_w, batch_size, capacity):
    # create input queues
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)

    # process path and string tensor into an image and a label
    file_content = tf.read_file(input_queue[0])
    # images = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    images = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
    images = tf.image.resize_image_with_crop_or_pad(images, IMAGE_HEIGHT, IMAGE_WIDTH)
    images = tf.image.per_image_standardization(images)

    labels = input_queue[1]

    image_one = tf.reshape(images, [-1], name=None)
    image_one.set_shape([image_h * image_w * NUM_CHANNELS])

    labels.set_shape([CLASSES_NUM])

    # collect batches of images before processing
    image_batch, label_batch = tf.train.batch(
        [image_one, labels],
        batch_size=batch_size,
        # num_threads=1,
        capacity=capacity
    )
    return image_batch, label_batch


def get_random_batch(image, label, image_h, image_w, batch_size, capacity):
    # create input queues
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)

    # process path and string tensor into an image and a label
    file_content = tf.read_file(input_queue[0])
    # images = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    images = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
    images = tf.image.resize_image_with_crop_or_pad(images, IMAGE_HEIGHT, IMAGE_WIDTH)
    images = tf.image.per_image_standardization(images)

    labels = input_queue[1]

    image_one = tf.reshape(images, [-1], name=None)
    image_one.set_shape([image_h * image_w * NUM_CHANNELS])

    labels.set_shape([CLASSES_NUM])

    # collect batches of images before processing
    image_batch, label_batch = tf.train.shuffle_batch(
        [image_one, labels],
        batch_size=batch_size,
        min_after_dequeue=MIN_DEQUEUE,
        # num_threads=1,
        capacity=capacity
    )
    return image_batch, label_batch


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 64, 64, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 16 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16], stddev=0.01)),

    # 7x7 conv, 16 inputs, 8 outputs
    'wc2': tf.Variable(tf.random_normal([7, 7, 16, 8], stddev=0.01)),

    # 5x5 conv, 8 inputs, 16 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 8, 16], stddev=0.01)),

    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([4 * 4 * 16, 100], stddev=0.01)),

    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([100, n_classes], stddev=0.01))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([8])),
    'bc3': tf.Variable(tf.random_normal([16])),
    'bd1': tf.Variable(tf.random_normal([100])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
print('pred.shape=', pred.shape)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

img_batch, lab_batch = get_batch(random_paths, random_labels, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZES, CAPACITY)
print(img_batch.shape)
print(lab_batch.shape)

# Testing
img_b, lab_b = get_random_batch(random_paths, random_labels, IMAGE_HEIGHT, IMAGE_WIDTH, 10, CAPACITY)
print(img_b.shape)
print(lab_b.shape)

saver = tf.train.Saver()

logfile = 'run.log'
print(logfile)
output = open(logfile, 'a+')
output.write('\n\n\n***************************************************************************\n')
timestamp = "\t\t" + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
output.write(timestamp)
output.flush()

with tf.Session() as sess:
    print('***************************************************************************')
    # initialize the variables
    sess.run(init)

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("from the train set: ---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    try:
        while not coord.should_stop():
            for epoch in range(0, train_epoch):
                for batch in range(0, batch_num):
                    img, lab = sess.run([img_batch, lab_batch])
                    # print(img.shape)
                    # print(lab.shape)
                    # print('y_=', lab)

                    # Run optimization op (backprop)
                    sess.run(optimizer, feed_dict={x: img, y: lab,
                                                   keep_prob: dropout})
                # Calculate loss and accuracy
                img1, lab1 = sess.run([img_b, lab_b])
                # print(img1.shape)
                # print(lab1.shape)

                loss, acc, corr, pre = sess.run([cost, accuracy, correct_pred, pred], feed_dict={x: img1,
                                                                                                 y: lab1,
                                                                                                 keep_prob: 1.0})
                # print('y_=', pre, 'y=', lab1, 'corr=', corr)
                print('corr=', corr)

                print("Epoch: " + str(epoch) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.6f}".format(acc))

                line = "Epoch: " + str(epoch) + ", Loss= " + \
                       "{:.6f}".format(loss) + ", Training Accuracy= " + \
                       "{:.6f}".format(acc)
                output.write(line + '\n')
                output.flush()

                if (epoch + 1) % 10 == 0:
                    print('<---save mode--->', epoch)
                    saver.save(sess, "./model/model.ckpt", epoch)

                if (loss < 0.00001) & (acc == 1.0):
                    coord.request_stop()

    except tf.errors.OutOfRangeError:
        output.close()
        print('done training --epoch limit reached!')
    finally:
        output.close()
        coord.request_stop()
        coord.join(threads)
sess.close()

