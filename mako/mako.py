import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor as Executor
import itertools
import os
import pkg_resources
import sys

import h5py
from fast5_research.fast5 import Fast5, iterate_fast5
import numpy as np

# Filtering the warnings doesn't appear to stop some tf warnings from showing,
#   so we'll wrap up stderr whilst we import keras.
from contextlib import contextmanager
@contextmanager
def silence_stderr():
    new_target = open(os.devnull, "w")
    old_target, sys.stderr = sys.stderr, new_target
    try:
        yield new_target
    finally:
        sys.stderr = old_target

with silence_stderr():
    from keras import layers
    from keras.models import Sequential
    from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
    from keras.utils import to_categorical
    from keras import optimizers


import logging

logger = logging.getLogger(__name__)
__mask_value__ = 1000.0 # shouldn't occur in data


def mako_resource(filename, subfolder='data'):
    return pkg_resources.resource_filename('mako',
        os.path.join(subfolder, filename))


def get_model(num_classes=2, cudnn=False, masking=False):
    """Build a model graph of a predefined structure.

    :param num_classes: number of output classes.
    :param cudnn: Use CuDNN layer.
    :param masking: Allow use of masking (to process sequences of varying length).

    """
    conv_filters = 96
    conv_window_size = 11
    stride = 5
    gru_size = 96

    grulayer = layers.CuDNNGRU if cudnn else layers.GRU
    logger.info('Building model with: {}.'.format(grulayer))
    model = Sequential()
    model.add(layers.Conv1D(
        conv_filters, conv_window_size, strides=stride,
        padding='same',
        input_shape=(None, 1)
    ))
    if masking:
        model.add(layers.Masking(mask_value=__mask_value__))
    for i, direction in enumerate(('rev', 'fwd', 'rev', 'fwd', 'rev')):
        if direction == 'rev':
            model.add(layers.Lambda(
            lambda x: x[:,::-1,:],
        ))
        model.add(grulayer(
            gru_size, return_sequences=True,
            name="gru_{}_{}".format(i, direction)
        ))
        if direction == 'rev':
            model.add(layers.Lambda(
            lambda x: x[:,::-1,:],
        ))
    model.add(grulayer(
        gru_size, return_sequences=False,
        name="gru_labeller"
    ))
    model.add(layers.Dense(
        num_classes, activation='softmax',
        name='classify'
    ))
    return model


def med_mad(data, factor=1.4826, axis=None):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median.
    
    :param data: A :class:`ndarray` object
    :param axis: For multidimensional arrays, which axis to calculate over 

    :returns: a tuple containing the median and MAD of the data

    .. note :: the default `factor` scales the MAD for asymptotically normal
        consistency as in R.
     
    """
    dmed = np.median(data, axis=axis)
    if axis is not None:
        dmed1 = np.expand_dims(dmed, axis)
    else:
        dmed1 = dmed
    
    dmad = factor * np.median(
        np.abs(data - dmed1),
        axis=axis
    )
    return dmed, dmad


def _scale_data(data):
    if data.ndim == 3:
        #(batches, timesteps, features)
        med, mad = med_mad(data, axis=1)
        med = med.reshape(med.shape + (1,))
        mad = mad.reshape(mad.shape + (1,))
        data = (data - med) / mad
    elif data.ndim == 1:
        med, mad = med_mad(data)
        data = (data - med) / mad
    else:
        raise AttributeError("'data' should have 3 or 1 dimensions.")
    return data


def _pad_and_scale(data, max_len):
    padded = np.full(
        (len(data), max_len, 1), __mask_value__, dtype=data[0].dtype
    )
    for i, r in enumerate(data):
        r = _scale_data(r)
        if len(r) > max_len:
            r = r[:max_len]
        padded[i, :len(r), 0] = r
    return padded


class Demultiplexer(object):
    def __init__(self, model_file):
        """Make predictions from input model and squiggle data.

        :param model_file: .hdf file from training.
        """
        with h5py.File(model_file, 'r') as h:
            try:
                attrs = h.attrs
                self.labels = [l.decode() for l in h.attrs['labels']]
                self.max_len = h.attrs['max_len']
                self.cudnn = h.attrs['cudnn']
            except:
                # Make some assumptions
                logger.info('Assuming model meta information.')
                nlabels = len(h['/model_weights/classify/classify/bias:0'])
                self.labels = ['barcode_{}'.format(x) for x in range(1, nlabels + 1)]
                self.max_len = 1500
                self.cudnn = False
                 
        self.caller = get_model(cudnn=self.cudnn, masking=True, num_classes=len(self.labels))
        self.caller.load_weights(model_file)


    def call_read(self, raw):
        """Call a single read.

        :param raw: 1D array containing raw data.
 
        """
        logger.info("Calling one read.")
        calls = self.call_many(raw[np.newaxis, :, np.newaxis])
        return calls[0]


    def call_many(self, raw, batch_size=500):
        """Call many reads.

        :param raw: 3D array in usual keras manner:
            (batches, timesteps, features), i.e (squiggle index, raw, 1)
            or a list of 1D arrays of variable length. In the latter case
            data will be padded and masked when given to GPU (this can lead
            to slightly different results), and network output will be trimmed
            back to appropriate region.
        :param batch_size: processing batch size.

        .. note:: data will be normalised per squiggle.

        """
        if isinstance(raw, np.ndarray):
            raw = _scale_data(raw)
        else:
            raw = _pad_and_scale(raw, self.max_len)
        results = self.caller.predict(raw, batch_size=batch_size)
        return results


def train(x_data, y_data, num_classes, train_name, batch_size=2000, epochs=5000, validation_split=0.05, cudnn=False, weights=None):
    opts = dict(verbose=1, save_best_only=True, mode='max')
    final_model = 'model.best.hdf5'
    callbacks = [
        # Best model according to training set accuracy
        ModelCheckpoint(
            os.path.join(train_name, 'model.best.train.hdf5'),
            monitor='acc', **opts),
        # Best model according to validation set accuracy
        ModelCheckpoint(
            os.path.join(train_name, final_model),
            monitor='val_acc', **opts),
        # Checkpoints when training set accuracy improves
        ModelCheckpoint(
            os.path.join(train_name, 'model-imp-{epoch:02d}-{acc:.2f}.hdf5'),
            monitor='acc', **opts),
        # Stop when no improvement
        EarlyStopping(monitor='val_loss', patience=20),
        # Log of epoch stats
        CSVLogger(os.path.join(train_name, 'training.log')),
    ]

    model = get_model(cudnn=cudnn, num_classes=num_classes)
    if weights is not None:
        model.load_weights(weights)
    model.compile(
       loss='categorical_crossentropy',
       optimizer=optimizers.Nadam(),
       metrics=['accuracy'],
    )
    model.fit(
        x_data, y_data,
        batch_size=batch_size, epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
    )
    return os.path.join(train_name, final_model)



def _fast5_filter(path, channels='all', recursive=False, limit=None, channel_limit=None):
    """Yield .fast5 filehandles, optionally filtered by channel.

    :param path: input path.
    :param channels: one of 'all', 'even', 'odd'.
    :param recursive: find .fast5 recursively below `path`.
    :param limit: maximum number of filehandles to yield.
    :param channel_limit: maximum number of filehandles to yield per channel.
    """
    allowed_channels = ('all', 'even', 'odd')
    if channels not in allowed_channels:
        raise ValueError(
            "'channels' option should be one of {}.".format(allowed_channels))

    def _odd_even_filter(base):
        for x in base:
            odd = bool(int(x.summary()['channel']) % 2)
            if (channels == 'odd' and odd) or (channels == 'even' and not odd):
                yield x

    def _chan_limit_filter(base):
        counters = Counter()
        for x in base:
            channel = int(x.summary()['channel'])
            if counters[channel] < channel_limit:
                counters[channel] += 1
                yield x

    def _valid_file(base):
        for fname in base:
            try:
                fh = Fast5(fname)
            except Exception as e:
                logger.warn('Could not open {}.'.format(fname))
            else:
                yield fh

    gen = _valid_file(iterate_fast5(path, paths=True, recursive=recursive))
    if channels != 'all':
        gen = _odd_even_filter(gen)
    if channel_limit is not None:
        gen = _chan_limit_filter(gen)
    if limit is not None:
        gen = itertools.islice(gen, limit)
    yield from gen


def load_data(datasets, max_len, channels='all', limit=None, channel_limit=None):
    """Load training datasets, from strings 'label:path'.

    :param datasets: strings or form 'label:path', path should contain .fast5
        files.
    :param max_len: max_len of training samples (longer squiggles will be
        truncated.
    :param channels: Squiggle channel filter, one of 'odd', 'even', or 'all'.
    :param limit: maximum number of examples per dataset.
    :param channel_limit: maximum number of examples per channel (per dataset).
    """
    x_train = []
    y_train = []
    labels = []
    for i, dataset in enumerate(datasets):
        label, path = dataset.split(":")
        logger.info("Reading dataset {} from {}, channels:{}, limit:{}".format(label, path, channels, limit))
        labels.append(label)
        raw = [x.get_read(raw=True) for x in _fast5_filter(path, channels, limit=limit, channel_limit=channel_limit)]
        if len(raw) == 0:
            raise RuntimeError("No data found for dataset {}.".format(dataset))
        logger.info("Got {} samples for '{}'.".format(len(raw), label))
        raw = _pad_and_scale(raw, max_len)
        x_train.append(raw)
        y_train.extend([i]*len(raw))

    x_train = np.concatenate(x_train)
    permutation = np.random.permutation(len(x_train))
    x_train = x_train[permutation]
    y_train = to_categorical(np.array(y_train)[permutation])

    logger.info('Loaded {} samples ({}).'.format(len(x_train), x_train.shape))
    return x_train, y_train, labels


def group(iterable, n):
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)



def main():
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)
    logging.basicConfig(format='[%(asctime)s - %(name)s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser(description='mako - short analyte tagger')

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--cudnn', action='store_true',
        help='Use cudnn layers rather than standard keras. This option must '
        'be set during inference with models trained with cuDNN layers.'
    )
    common.add_argument('--channels', default='all',
        choices=['all', 'odd', 'even'],
        help='filter reads by channel.')

    subparsers = parser.add_subparsers(title='subcommands', description='valid commands',
        help='additional help', dest='command')
    subparsers.required = True
  
    training = subparsers.add_parser('train', help='Train a basecaller.', parents=[common])
    training.add_argument('datasets', nargs='+', help="Datasets in form 'label:folder'.")
    training.add_argument('--dataset', help='Save/load prepared dataset.')
    training.add_argument('--weights', help='Initial weights (perhaps from previous training).')
    training.add_argument('--output', default='training', help='Output folder.')
    training.add_argument('--max_len', type=int, default=1500, help='Maximum signal length.')
    training.add_argument('--limit', default=None, type=int,
        help='Maximum number of examples per class.')
    training.add_argument('--channel_limit', default=None, type=int,
        help='Maximum number of examples per channel (per class).')
  
    predict = subparsers.add_parser('predict', help='Basecall .fast5 reads.', parents=[common])
    predict.add_argument('input',
        help="Path to input .fast5s, will be searched recursively.")
    predict.add_argument('output',
        help='Output file.')
    predict.add_argument('--batch_size', type=int, default=5000,
        help='Number of reads to process in a batch.')
    predict.add_argument('--model_file', default=mako_resource('default_model.hdf'),
        help='Model file from training.')
    predict.add_argument('--limit', default=None, type=int,
        help='Number of input examples to process.')
    predict.add_argument('--pass_filter', default=0.256, type=float,
        help="Simple filter on classification entropy, reads not meeting this "
             "filter will be tagged as pass=0 in output summary. The default "
             "value is chosen to give ~0.25% classification error. Setting a "
             "smaller value will decrease error at the expense of discarding "
             "data.")

    args = parser.parse_args()
    commands = ('train', 'predict')
    if args.command not in commands:
        raise ValueError("'command' argument must be one of {}.".format(commands))

    if args.command == 'train':
        if os.path.isdir(args.output):
            raise RuntimeError("Output directory already exists, refusing to overwrite.")
        os.mkdir(args.output)
        if args.dataset is not None and os.path.isfile(args.dataset):
            logger.info('Loading pre-created dataset {}.'.format(args.dataset))
            with h5py.File(args.dataset, 'r') as h:
                x, y = h['x'][()], h['y'][()]
                labels = [l.decode() for l in h.attrs['labels']]
        else:
            logger.info('Loading datasets.')
            x, y, labels = load_data(args.datasets, args.max_len,
                channels=args.channels, limit=args.limit, channel_limit=args.channel_limit)
            if args.dataset is not None:
                logger.info('Saving dataset to {}.'.format(args.dataset))
                with h5py.File(args.dataset, 'w') as h:
                    h['x'] = x
                    h['y'] = y
                    h.attrs['labels'] = np.array([l.encode() for l in labels])

        # run training and save model
        final_model = train(x, y, len(labels),
            args.output, cudnn=args.cudnn, weights=args.weights)
        logger.info('Appending meta info to final model: {}.'.format(final_model))
        with h5py.File(final_model, 'a') as h:
            h.attrs['labels'] = np.array([l.encode() for l in labels])
            h.attrs['max_len'] = args.max_len
            h.attrs['cudnn'] = args.cudnn

    elif args.command == 'predict':
        def data_from_fh(fh):
            try:
                raw = fh.get_read(raw=True)
                summary = fh.summary()
                med, mad = med_mad(raw)
                summary['median_current'] = med
                summary['stdv_current'] = mad
            except:
                logging.warn("Error extracting data from {}.".format(fh.filename))
                return None
            else:
                return raw, summary

        def batcher():
            data = (
                data_from_fh(x)
                for x in _fast5_filter(args.input, limit=args.limit, channels=args.channels, recursive=True)
            )
            filtered = (x for x in data if x is not None)
            yield from group(filtered, args.batch_size)


        # determine summary data fields
        try:
            _, peek = next(next(batcher()))
        except StopIteration:
            logger.error('No valid files found.')
            sys.exit(1)
        summary_keys = list(peek.keys())
        for k in ('filename', 'read_id'):
            try:
                summary_keys.remove(k)
            except:
                pass
            else:
                summary_keys.insert(0, k)


        caller = Demultiplexer(model_file=args.model_file)
        with open(args.output, 'w') as output:
            output.write('\t'.join(
                summary_keys +
                ['classification', 'score', 'pass', 'entropy'] +
                ['score_{}'.format(x) for x in caller.labels]
            ))
            output.write("\n")
            for i, batch in enumerate(batcher(), 1):
                logger.info('Processing batch {}.'.format(i))
                logger.info('Reading data...') 
                squiggles, summaries = zip(*list(batch))
                logger.info('Performing classification...')
                for summary, results  in zip(summaries, caller.call_many(squiggles)):
                    entropy = -np.sum(np.dot(results, np.log(results)))
                    best = np.argmax(results)
                    best_score = results[best]
                    best_label = caller.labels[best]
                    passed = entropy < args.pass_filter
                    summary_values = [summary[k] for k in summary_keys]
                    output.write('\t'.join(str(x)
                        for x in (*summary_values, best_label, best_score, int(passed), entropy, *results)
                    ))
                    output.write("\n")


if __name__ == '__main__':
    main()
