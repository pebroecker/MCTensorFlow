import sys
import os
import numpy as np
import tensorflow as tf
import data_loader as loader
import convolutional_softmax as model
import h5py

print (os.getcwd())

# training parameters
paths = sys.argv[1].split(",")  # paths in HDF5 files
parameter = sys.argv[2]
file_pattern = sys.argv[3]      # file pattern for prediction 1
checkpoint = sys.argv[4]        # path for saving results
batch_size = 128
n_epochs = 128

# if not os.path.exists(checkpoint):
#     print("Checkpoint not found")
#     sys.exit(0)

with h5py.File(file_pattern.format(i=1), "r") as h5file:
    batch_size = h5file["{}/1".format(paths[0])][...].shape[0]
    image_size = h5file["{}/1".format(paths[0])][...].shape[1]

print("Image size is {}".format(image_size))

x = tf.placeholder(tf.float32, [None, image_size, image_size, len(paths)])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

m = model.convolutional_softmax(x, 3, 32, 512, keep_prob)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint)

    global_step = tf.get_collection_ref("global_step")
    print(global_step)
    gs = sess.run(global_step)
    print("Global step ", gs)

    for task in range(1, 100):
        if not os.path.exists(file_pattern.format(i=task)):
            continue
        try:
            with h5py.File(file_pattern.format(i=task), "r") as h5file:
                accuracies = []

                h = h5file[parameter][...]
                input_x = np.zeros((batch_size, image_size, image_size, len(paths)))
                input_y = np.zeros((batch_size, 2))

                for k in h5file[paths[0]].keys():
                    for ip, p in enumerate(paths):
                        input_x[:, :, :, ip] = h5file["{}/{}".format(p, k)][...]

                    result = m.prediction.eval(session=sess, feed_dict={x: input_x, y: input_y, keep_prob:1.0})
                    accuracies.append(np.mean(result, 0)[1])

                print("The result is", h, np.mean(accuracies))
        except KeyboardInterrupt:
            raise
        except:
            print(sys.exc_info())
            pass
