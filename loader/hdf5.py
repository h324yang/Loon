import h5py
import numpy as np
from collections import defaultdict


class HDF5er:
    """
    (HDF5er) is designed for training on large files. HDF5 format is used.
    By transfroming raw data into HDF5 datasets, we can bucketize such data to derive minibatches,
    which can be stored in one another HDF5 file.
    data_path: file path of raw data. (str)
    fields: field names in order. (list of str)
    tokenizer: tokenizing function. (func: str -> list of int)
    preprocess: applied to the tokens after tokenization. (func: list of int -> list of int)
    max_length: the max sequence lengths corresponding to fields. (list of int)
    delimiter: used for splitting raw data. (char)
    skip: the fields skipped and will be stored as int. (list of bool)
    """

    def __init__(self, data_path, fields, tokenizer, preprocess, max_length, delimiter="\t", skip=None):
        assert len(fields) == len(max_length), "number of fields and number of max_length are not the same."
        self.data_path = data_path
        self.fields = fields
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = max_length
        self.delimiter = delimiter
        self.skip = skip

    def cutting_or_padding(self, array, max_length):
        cur_length = len(array)
        if cur_length >= max_length:
            return np.array(array)[:max_length]
        else:
            return np.pad(array, (0, max_length - cur_length), mode='constant')

    def add_instance_to_hdf5(self, dataset, array):
        last, dim = dataset.shape
        dataset.resize((last + 1, dim))
        dataset[last] = self.cutting_or_padding(array, dim)

    def save_to_hdf5(self, hdf5_file, verbose=False):
        self.errors = []
        with h5py.File(hdf5_file, "w") as file:
            # create dataset
            for field, length in zip(self.fields, self.max_length):
                file.create_dataset(field, shape=(0, length), maxshape=(None, None), chunks=True)
                file.create_dataset(field + "_len", shape=(0, 1), maxshape=(None, None), chunks=True)

            for i, e in enumerate(self.get_instance_from_dataset()):
                if verbose:
                    print("Inserting record %d..."%i)
                is_complete = True
                for field in self.fields:  # check every field is successfully processed
                    if not field in e:
                        is_complete = False
                if is_complete:
                    for field in self.fields:
                        data = e[field]
                        cur_len = len(data)
                        self.add_instance_to_hdf5(file[field], data)
                        self.add_instance_to_hdf5(file[field + "_len"], np.array([cur_len]))  # len of each is stored
                else:
                    self.errors.append(i)

    def process(self, text):
        transformed = self.tokenizer(text)
        if self.preprocess:
            transformed = self.preprocess(transformed)
        return transformed

    def get_instance_from_dataset(self):
        with open(self.data_path) as f:
            for line in f:
                if self.skip:
                    yield {k: self.process(v) if not skip else [int(v)] for skip, k, v in zip(self.skip, self.fields, line.strip().split(self.delimiter))}
                else:
                    yield {k: self.process(v) for k, v in zip(self.fields, line.strip().split(self.delimiter))}

    def get_bucket(self, sent_len):
        for i, (s, e) in enumerate(self.buckets):
            if s <= sent_len < e:
                return self.buckets[i]

    def bucketize(self, hdf5_file, buckets):
        self.buckets = buckets  # list of tuples, which indicate bucket boundaries
        self.weights = {}
        self.total_weight = 0.
        self.by_bucket = defaultdict(list)
        with h5py.File(hdf5_file) as file:
            self.data_size = file[self.fields[0]].shape[0]
            for i in range(self.data_size):
                example_len = sum(file[k + "_len"][i] for k in self.fields)
                self.by_bucket[self.get_bucket(example_len)].append(i)
                self.weights[i] = np.sqrt(example_len)
                self.total_weight += self.weights[i]

    def get_raw_minibatches(self, batch_size):
        self.batch_size = batch_size
        weight_per_batch = self.batch_size * self.total_weight / self.data_size
        cumulative_weight = 0.0
        id_batches = []
        for _, ids in self.by_bucket.items():
            ids = np.array(ids)
            np.random.shuffle(ids)
            curr_batch, curr_weight = [], 0.0
            for i, curr_id in enumerate(ids):
                curr_batch.append(curr_id)
                curr_weight += self.weights[curr_id]
                if (i == len(ids) - 1 or
                        cumulative_weight + curr_weight >= (len(id_batches) + 1) * weight_per_batch):
                    cumulative_weight += curr_weight
                    id_batches.append(np.array(curr_batch))
                    curr_batch, curr_weight = [], 0.0
        np.random.shuffle(id_batches)
        return id_batches

    def save_batches(self, hdf5_source, hdf5_target, batch_size, buckets=None):
        if not buckets: buckets = [(0, 1e5)]
        print("bucketizing...")
        self.bucketize(hdf5_source, buckets)
        print("sampling batches...")
        batches = self.get_raw_minibatches(batch_size)
        with h5py.File(hdf5_source, "r") as source:
            with h5py.File(hdf5_target, "w") as file:
                # create dataset
                file.create_dataset("minibatch_size", shape=(0, 1), maxshape=(None, None), chunks=True)
                for field, length in zip(self.fields, self.max_length):
                    file.create_dataset(field, shape=(0, length), maxshape=(None, None), chunks=True)
                    file.create_dataset(field + "_len", shape=(0, 1), maxshape=(None, None), chunks=True)
                    file.create_dataset(field + "_minibatch_max_length", shape=(0, 1), maxshape=(None, None),
                                        chunks=True)

                for bid, batch in enumerate(batches):
                    print("Inserting batch {}...".format(bid))
                    minibatch_size = len(batch)
                    self.add_instance_to_hdf5(file["minibatch_size"], np.array([minibatch_size]))
                    for field in self.fields:
                        minibatch_max_length = max(source[field + "_len"][i] for i in batch)
                        self.add_instance_to_hdf5(file[field + "_minibatch_max_length"],
                                                  np.array([minibatch_max_length]))
                        for i in batch:
                            self.add_instance_to_hdf5(file[field], source[field][i])
                            self.add_instance_to_hdf5(file[field + "_len"], source[field + "_len"][i])

    def get_minibatches(self, hdf5_file):
        return Minibatch(hdf5_file, self.fields)


class Minibatch:
    def __init__(self, hdf5_file, fields):
        self.hdf5_file = hdf5_file
        self.fields = fields
        with h5py.File(self.hdf5_file, "r") as file:
            self.minibatch_size = file["minibatch_size"][:]
            self.total_num_batch = file["minibatch_size"].shape[0]

    def __iter__(self):
        self.head = 0
        with h5py.File(self.hdf5_file, "r") as file:
            for bid, size in enumerate(self.minibatch_size):
                size = int(size)
                minibatch = []
                for field in self.fields:
                    max_length = int(file[field + "_minibatch_max_length"][bid])
                    minibatch.append(file[field][self.head:self.head + size][:, :max_length])
                self.head += size
                yield minibatch

    def __len__(self):
        return self.total_num_batch




