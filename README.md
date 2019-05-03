LPagg
=====
The load profile aggregator combines profiles for heat and power demand
of buildings from different sources.

Installation
============
1. You need Python. The recommended way is to install Python with
[Anaconda](https://www.anaconda.com/distribution/),
a package manager that distributes Python with data science packages.
During installation, (despite the warning) please set the advanced option:
```
[x] Add Anaconda to my PATH environment variable
```

2. You also need to install [Git](https://git-scm.com/downloads) for
downloading this repository.

3. Then you can clone this repository to a directory of your choice by
opening a `cmd` window and writing:
```
git clone https://github.com/jnettels/lpagg.git
```
4. Now you need to change directory into the new folder:
```
cd lpagg
```
5. From here you can build and install `lpagg` with conda:

```
conda build conda.recipe
conda install --use-local lpagg -y
```

Update
------
When an update to `lpagg` is available in this repository, you can simply
change to the folder from step 4 and download the latest files with:
```
git pull
```
Afterwards, repeat step 5 to build and install the update.

Usage
=====
lpagg
-----
You should be able to start the program with a `cmd` window:
```
python lpagg
```
This will bring up a file dialog for choosing a YAML configuration file
that contains all the settings required for the program. To try it,
you can choose the example `lpagg\examples\VDI_4655_config_example.yaml`.

You can also show a help message:
```
python lpagg --help
```
Another approach is to place a shortcut where you would like to use it.
Moreover, you can now write you own Python scripts that use `lpagg`.
Use the script `__main__.py` in this repository as an example.

simultaneity
------------
One feature of lpagg is creating the effects of a simultaneity factor.
Copies of a given time series are created and, if a standard deviation
``sigma`` is given, a time shift is applied to the copies.
This can also be used as a standalone script, where you have to
provide a file with time series data. In a `cmd` window, write the
following to learn more:
```
python simultaneity.py --help
```
