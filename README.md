Mako
====

`mako` is a software package for identification of analyte panels from 
nanopore signal data, as contrasted with approaches that perform similar
tasks after basecalling. It leverages the keras neural networks API to
perform inference.


Scope
-----

The software is provided as a simple demonstration of how a developer might
start on a path of creating a similar system with high precision and recall
from signal data, rather than basecalled data as most extant systems do. It is
***not*** intended to be a replacement for these latter tools, at least
currently.

***Very little effort has been spent on optimising the recall and
precision of mako, it can undoubtedly be improved.***

Community development is encouraged, see Development below.


Installation
------------

Mako is written in Python 3 and can be installed in a standard python manner.
From the source directory simply run:

    python setup.py install


Usage
-----

`mako` comes with two core commands: `mako train` and `make predict`, for
training and using an inference model respectively. Input data is assumed
to be `.fast5` read files as output by Oxford Nanopore Technologies'
devices.

Basic training requires separating input data for distinct analytes into
distinct directories. The training program will then accept these as:

    mako train <label_0>:<folder_0> <label_1>:<folder_1> ...

where `<label_x>` and `<folder_x>` may be replaced as fitting. The labels are
used only in reporting for `mako predict`. When training has finished, the path
to the trained model will be reported, this can be given to `mako predict`.

Identification of unknown data ("tagging") is performed using `mako predict`. The
program requires only a path to input data, and optionally a trained model:

    mako predict <input_folder> <output_file> --model_file <from mako train>

The output file is a tab-separated text file containing predictions of
identification of reads, along with summary statistics of reads.


Outline of current inference scheme
-----------------------------------

`mako` currently uses a similar RNN to that used within basecallers. The major
modication is that a single output variable is produced for an input sequence,
rather than a sequence of outputs as in a basecaller. `mako` also only uses the
beginning of a read for identification, though this is easily changed.


Example training data
---------------------

To aid development an example dataset of 12 analytes is available for download
here:
[mako_test_data](https://s3-eu-west-1.amazonaws.com/ont-research/mako_test_data.tar.gz).



Development
-----------

Community development of `mako` is encouraged, please submit issues and PRs on
github. There is plenty of scope for improvement:

* Changes to neural network
* Preprocessing the inputs
* Generalisations to arbitrary use cases
* ...

