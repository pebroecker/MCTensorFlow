from __future__ import print_function
import h5py
import numpy as np
import numpy.random as npr
import tensorflow as tf
import sys

class loader:

    def __init__(self, paths, files, mini_batch_size, n_train_ids, n_test_ids):
        self.mini_batch_size = mini_batch_size
        self.paths = paths
        self.files = files
        self.n_train_ids = n_train_ids
        self.n_test_ids = n_test_ids
        self.current_train_id = 0
        self.current_test_id = 0

        with h5py.File(files[0][0], "r") as h:
            batch_ids = sorted([int(k) for k in h[paths[0]].keys()])
            self.train_ids = batch_ids[0:n_train_ids]
            if n_test_ids == -1:
                self.test_ids = batch_ids[n_train_ids:]
                self.n_test_ids = len(self.test_ids)
            else:
                self.test_ids = batch_ids[n_train_ids:n_train_ids + n_test_ids]

            self.id_size = h["{path}/{id}".format(path=paths[0], id=self.train_ids[0])].shape[0]
            self.image_size = h["{path}/{id}".format(path=paths[0], id=self.train_ids[0])].shape[1]

        self.n_points = len(files)
        print("Working with ", self.n_points, " training points")
        self.n_samples = self.n_points * n_train_ids * self.id_size
        assert(self.n_samples % mini_batch_size == 0), "Number of samples is not a multiple of mini batch size"

        self.n_runs_per_epoch = int(self.n_samples / self.id_size)
        print("There are {0} runs per epoch".format(self.n_runs_per_epoch))
        if self.n_samples % mini_batch_size != 0:
            print("Sample size is not a multiple of mini batch size")
            sys.exit(0)

        self.x = np.zeros((self.n_samples, self.image_size, self.image_size, len(paths)), dtype=np.float32)
        self.y = np.zeros((self.n_samples, 2), dtype=np.float32)

        idx = 0

        print("Loading all training samples")
        for f in files:
            for i in self.train_ids:
                with h5py.File(f[0], "r") as h:
                    for pi, p in enumerate(paths):
                        self.x[idx * self.id_size: (idx + 1) * self.id_size, :, :, pi] = h["{path}/{i}".format(path=p, i=i)][...]
                    self.y[idx * self.id_size: (idx + 1) * self.id_size, f[1]] = 1
                    idx += 1

        if idx * self.id_size != self.n_samples:
            print(idx * self.id_size, self.n_samples)
            print("Not all training samples were loaded as expected.")
            sys.exit(0)

        self.n_test_samples = self.n_points * self.n_test_ids * self.id_size
        self.n_runs_per_evaluation = int(self.n_test_samples / self.id_size)
        self.train_id_order = npr.random_integers(0, self.n_samples - 1, self.n_samples)
        self.x_test = np.zeros((self.n_test_samples, self.image_size, self.image_size, len(paths)), dtype=np.float32)
        self.y_test = np.zeros((self.n_test_samples, 2), dtype=np.float32)

        idx = 0

        print("Loading all test samples")
        for f in files:
            for i in self.test_ids:
                with h5py.File(f[0], "r") as h:
                    for ip, p in enumerate(paths):
                        self.x_test[idx * self.id_size: (idx + 1) * self.id_size, :, :, ip] = h["{path}/{i}".format(path=p, i=i)][...]
                    self.y_test[idx * self.id_size: (idx + 1) * self.id_size, f[1]] = 1
                    idx += 1

        if idx * self.id_size != self.n_test_samples:
            print("Not all test samples were loaded as expected.")
            sys.exit(0)
        self.current_test_id = 0

        self.current_x = np.zeros((mini_batch_size, self.image_size, self.image_size, len(paths)), dtype=np.float32)
        self.current_y = np.zeros((mini_batch_size, 2), dtype=np.float32)

        print("Memory usage of data_loader.py")
        print("=================================")
        print(self.x.nbytes / (1000 * 1000), " MB")
        print(self.y.nbytes / (1000 * 1000), " MB")
        print(self.x_test.nbytes / (1000 * 1000), " MB")
        print(self.y_test.nbytes / (1000 * 1000), " MB")
        print(self.current_x.nbytes / (1000 * 1000), " MB")
        print(self.current_y.nbytes / (1000 * 1000), " MB")

    def next_train_batch(self):
        if self.current_train_id == int(self.n_samples / self.mini_batch_size):
            self.train_id_order = npr.random_integers(0, self.n_samples - 1, self.n_samples)
            self.current_train_id = 0

        train_ids = npr.random_integers(0, self.n_samples - 1, self.mini_batch_size)
        self.current_x[:, :] = self.x[train_ids, :]
        self.current_y[:, :] = self.y[train_ids, :]

        self.current_train_id += 1
        return self.current_x, self.current_y


    def next_test_batch(self):
        if self.current_test_id == int(self.n_test_samples / self.id_size):
            self.current_test_id = 0

        test_x = self.x_test[self.current_test_id * self.id_size:(self.current_test_id + 1) * self.id_size, :, :, :]
        test_y = self.y_test[self.current_test_id * self.id_size:(self.current_test_id + 1) * self.id_size, :]
        self.current_test_id += 1

        return test_x, test_y
