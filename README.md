Mako
====

Mako is a software package for squiggle tagging. It can be used to train a
model to recognise small panels of analytes from their squiggles.

Installation
------------

Mako is written in python, as such you will need a working python installation,
The following instructions can be followed to install and run the software on
Windows. (Users of other operating systems can likely skip to step 3.). 

1. Download and install python 3.6 (64bit):
   https://www.python.org/ftp/python/3.6.4/python-3.6.4-amd64.exe
2. Run the above, checking any and all checkboxes through the installation,
   explicitly:
   i.      Install launcher for all users
   ii.      Add Python 3.6 to PATH
   iii.      Customize installation
   iv.      Tick everything, then next
   v.      Click everything, then next
3. Download and unpack the software, open a command prompt and run:

    cd <where the software is located>    
    pip install keras fast5_research tensorflow --user
    pip install .
    mako predict path/to/reads output.txt
