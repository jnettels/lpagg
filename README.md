LPagg
=====
The load profile aggregator combines profiles for heat and power demand
of buildings from different sources. Data for households is taken from
VDI 4655<sup id="a1">[1](#f1)</sup>, while BDEW<sup id="a2">[2](#f2)</sup>
is used for commercial buildings. Both sources provide 24h profiles in
resolutions of 15 minutes for defined typical-days, which LPagg combines
to construct a selected calendar year. It takes into account input weather
data (from DWD<sup id="a3">[3](#f3)</sup>), local holidays and daylight
saving time. Moreover, a random time shift derived from a normal distribution
can be applied to each building, in order to approximate the effects of
simultaneity present in larger groups of buildings. While the sources
only represent very generalized load profiles, LPagg still provides a
fast and easy method to create reliable input data for annual simulations
of district energy systems. All settings have to be provided via a YAML
configuration file. LPagg was created as part of the publicly funded
project futureSuN<sup id="a4">[4](#f4)</sup>.

Installation
============

Installation with Anaconda
-----------------------
LPagg is a Python package. The recommended way to install the latest release
is by using [Anaconda](https://www.anaconda.com/distribution/):
```
conda install lpagg -c jnettels
```
In case of package conflicts, this might work instead:
```
conda install lpagg -c jnettels -c conda-forge
```

Installation from source
-----------------------
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
You should be able to start the program from a `cmd` window:
```
lpagg
```
This will bring up a file dialog for choosing a YAML configuration file
that contains all the settings required for the program. To try it,
you can choose the example [`lpagg\examples\VDI_4655_config_example.yaml`](https://github.com/jnettels/lpagg/blob/master/lpagg/examples/VDI_4655_config_example.yaml).

You can also show a help message:
```
lpagg --help
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
simultaneity --help
```
There is a version with a graphical user interface, which can be
started with the command:
```
simlty_GUI
```

Changelog
========
At the moment, no dedicated changelog is maintained. However, important
changes are noted on the
[release page](https://github.com/jnettels/lpagg/releases).



Literature
==========

|     |     | lpagg            | lpagg + demandlib    | demandlib            |
| --- | --- | ---------------- | -------------------- | -------------------- |
| HH  | RH  | VDI 4655 (lapgg) | VDI 4655 (demandlib) | VDI 4655 (demandlib) |
|     | TWE | VDI 4655 (lapgg) | VDI 4655 (demandlib) | VDI 4655 (demandlib) |
|     | EL  | VDI 4655 (lapgg) | VDI 4655 (demandlib) | VDI 4655 (demandlib) |
| GHD | RH  | futuresolar      | futuresolar          | BDEW-Formel          |
|     | TWE | DOE              | DOE                  | \-                   |
|     | EL  | BDEW-TT          | BDEW-TT              | BDEW-TT              |

<b id="f1">1</b> VDI 4655, 2008: Referenzlastprofile von Ein- und
Mehrfamilienhäusern für den Einsatz von KWK-Anlagen. [↩](#a1)

<b id="f2">2</b> BDEW (1999): Repräsentative VDEW-Lastprofile.
Unter Mitarbeit von BTU Cottbus. Frankfurt am Main. Online verfügbar unter
https://www.bdew.de/media/documents/1999_Repraesentative-VDEW-Lastprofile.pdf
[↩](#a2)

<b id="f3">3</b> Deutscher Wetterdienst (2017): Ortsgenaue Testreferenzjahre
von Deutschland für mittlere und extreme Witterungsverhältnisse. Handbuch.
Unter Mitarbeit von Bundesamt für Bauwesen und Raumordnung (BBR). Offenbach.
Online verfügbar unter
http://www.bbsr.bund.de/BBSR/DE/FP/ZB/Auftragsforschung/5EnergieKlimaBauen/2013/testreferenzjahre/try-handbuch.pdf
[↩](#a3)

<b id="f4">4</b> Bonk, Natalie; Juschka, Winfried; Kofler, Philipp;
Nettelstroth, Joris; Pröll, Markus; Bestenlehner, Dominik et al. (2020):
futureSuN. Analyse, Bewertung und Entwicklung zukunftsfähiger Anlagenkonzepte
für solare Nahwärmeanlagen mit saisonaler Wärmespeicherung. SIZ energie+,
SIZ EGS, IGTE, ZAE Bayern. Braunschweig, Stuttgart, München. Online verfügbar
unter https://siz-energie-plus.de/projekte/futuresun. [↩](#a4)
