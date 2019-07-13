from model.tokenization import BertTokenizer
from loader.hdf5 import HDF5er
from utils import never_split

vocab_path = "vocabs/tacred-bert-base-cased-vocab.txt"
TOKENIZER = BertTokenizer(vocab_file=vocab_path, never_split=never_split)

if __name__ == "__main__":
    # Generate train in hdf5
    debug_triple = "debug_dataset/triples.debug.tsv"
    debug_train_hdf5 = "debug_dataset/triples.debug.hdf5"
    debug_train_batch_hdf5 = "debug_dataset/triples.debug.batch.hdf5"
    batch_size = 2
    fields = ["query", "pos", "neg"]
    max_length = [64, 256, 256]
    add_special = lambda tokens: [101] + tokens + [102]  # [CLS]:101, [SEP]:102
    tokenize = lambda text: TOKENIZER.convert_tokens_to_ids(TOKENIZER.tokenize(text)[0])
    buckets = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, sum(max_length) + 1)]
    triplet_hdf5er = HDF5er(data_path=debug_triple, fields=fields, max_length=max_length,
                            tokenizer=tokenize, preprocess=add_special)
    triplet_hdf5er.save_to_hdf5(debug_train_hdf5, verbose=True)
    triplet_hdf5er.save_batches(hdf5_source=debug_train_hdf5, hdf5_target=debug_train_batch_hdf5, batch_size=batch_size)

    # Generate dev in hdf5
    debug_sent = "debug_dataset/sent.debug.tsv"
    debug_test_hdf5 = "debug_dataset/sent.debug.hdf5"
    debug_test_batch_hdf5 = "debug_dataset/sent.debug.batch.hdf5"
    fields = ["type", "id", "sent"]
    max_length = [1, 1, 256]
    skip = [True, True, False]
    batch_size = 2
    buckets = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, sum(max_length) + 1)]
    dev_hdf5er = HDF5er(data_path=debug_sent, fields=fields, max_length=max_length,
                        tokenizer=tokenize, preprocess=add_special, skip=skip)
    dev_hdf5er.save_to_hdf5(debug_test_hdf5, verbose=True)
    dev_hdf5er.save_batches(hdf5_source=debug_test_hdf5, hdf5_target=debug_test_batch_hdf5, batch_size=batch_size)


