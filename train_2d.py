import sys
import os
import numpy as np
import tensorflow as tf
import data_loader as loader
import convolutional_softmax as model
from copy import copy
from datetime import datetime


print (os.getcwd())
print(sys.argv[5])
# training parameters
alpha = float(sys.argv[1])
paths = sys.argv[2].split(",") # paths in HDF5 files
files = sys.argv[3].split(",") # files for class 1
left_files = copy(files)
left_labels = [0] * len(files)
files = sys.argv[4].split(",") # files for class 2
right_files = copy(files)
right_labels =  [1] * len(files)
base_path = sys.argv[5] # path for saving results
batch_size = 128
n_epochs = 32

# create necessary paths if not exist yet
if not os.path.isdir(base_path): os.mkdir(base_path)
checkpoint_dir = os.path.join(base_path, "checkpoints")
if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
tensorboard_dir = os.path.join(base_path, "tensorboard")
if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)

# set up pipeline for batches
l = loader.loader(paths, list(zip(left_files + right_files, left_labels + right_labels)), batch_size, 16, 8)

# TF placeholders for input x, output y
# and the retention rate for neurons in the last layer
x = tf.placeholder(tf.float32, [None, l.image_size, l.image_size, len(paths)])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(tf.constant(0), trainable=False, name="global_step")
tf.add_to_collection("global_step", global_step)
increment_global_step_op = tf.assign(global_step, global_step+1)

# initialize model and create necessary TF optimization operations
m = model.convolutional_softmax(x, 3, 32, 512, keep_prob)
prediction = m.softmax_linear

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y, name="cross_entropy_per_sample")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")

with tf.name_scope("train"):
    opt = tf.train.AdamOptimizer(alpha)
    grads = opt.compute_gradients(cross_entropy_mean)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# linking to tensorboard
for gradient, var in grads:
    tf.summary.histogram(var.name + "/gradient", gradient)

for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

tf.summary.scalar("cross_entropy", cross_entropy_mean)
# tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

# TF output writer
writer = tf.summary.FileWriter(tensorboard_dir)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("{} Starting training ...".format(datetime.now()))

    for epoch in range(n_epochs):
        for r in range(l.n_runs_per_epoch):
            train_x, train_y = l.next_train_batch()
            sess.run(apply_gradient_op, feed_dict={x: train_x, y: train_y, keep_prob:0.5})
            sess.run(increment_global_step_op)

            if r % 10 == 0 and r > 0:
                print("summary.merge_all - {} / {}".format(r, l.n_runs_per_epoch))
                s = sess.run(merged_summary, feed_dict={x: train_x,
                                                        y: train_y,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch * l.n_runs_per_epoch + r)

        print("{} Starting validation ...".format(datetime.now()))

        for t in range(l.n_runs_per_evaluation):
            l.current_test_id = 0
            test_x, test_y = l.next_test_batch()
            sess.run(m.prediction, feed_dict={x: train_x, y: train_y, keep_prob:1.})
            acc = accuracy.eval(session=sess, feed_dict={x: train_x, y: train_y, keep_prob:1.})

            l.current_test_id = int(l.n_test_samples / l.id_size) - 1
            test_x, test_y = l.next_test_batch()
            sess.run(m.prediction, feed_dict={x: train_x, y: train_y, keep_prob:1.})
            acc += accuracy.eval(session=sess, feed_dict={x: train_x, y: train_y, keep_prob:1.})

            l.current_test_id = 0
            print("{} Validation accuracy = {:.4f}".format(datetime.now(), acc / 2.))
            break

        print("{} Saving checkpoint of model.".format(datetime.now()))
        checkpoint_file = os.path.join(checkpoint_dir, "epoch_{}.ckpt".format(epoch + 1))
        save_path = saver.save(sess, checkpoint_file)
        print("{} Checkpoint saved at {}".format(datetime.now(), checkpoint_file))
