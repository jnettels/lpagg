LPagg
=====
The load profile aggregator combines profiles for heat and power demand
of buildings from different sources.

Installation
============
1. You need Python. The recommended way is to install Python with
[Anaconda](https://www.anaconda.com/distribution/),
a package manager that distributes Python with data science packages.
You also need to install [Git](https://git-scm.com/downloads) for
downloading this repository.

2. Then you can clone this repository to a directory of your choice by
opening a `cmd` window and writing:
```
git clone https://github.com/jnettels/lpagg.git
```
3. Now you need to change directory into the created folder:
```
cd lpagg
```
4. From here we can build and install `lpagg`:

```
conda build conda.recipe
conda install --use-local lpagg -y
```

Update
------
When an update to `lpagg` is available in this repository, you can simply
change to the folder from step 3. and download the latest files with
```
git pull
```
Afterwards, repeat step 4. build and install the updated files.

Usage
=====

You should be able to start the program with the command
`python lpagg` which will bring up a file dialog for choosing
a configuration file. You can also show a help message:
```
python lpagg --help
```