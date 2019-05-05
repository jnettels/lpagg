# -*- coding: utf-8 -*-
'''
**LPagg: Load profile aggregator for building simulations**

Copyright (C) 2019 Joris Nettelstroth

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.


LPagg
=====
The load profile aggregator combines profiles for heat and power demand
of buildings from different sources.


Module simultaneity GUI
-----------------------
Graphical user interface for the standalone ``simulataneity`` module.
'''
import numpy as np
import os
import sys
from PyQt5 import QtWidgets, Qt, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib as mpl

import lpagg.simultaneity


class MainClassAsGUI(QtWidgets.QMainWindow):
    '''Main GUI Window.
    CentralWidget consists of a splitter, to show the settings on the left
    and a figure on the right.
    '''
    def __init__(self):
        super().__init__()

        self.splitter = QtWidgets.QSplitter()
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)
        self.statusBar()

        self.plotWidget = MyMplCanvas()
        self.leftSideWidget = SettingsGUI(self.plotWidget, self.statusBar)
        self.splitter.addWidget(self.leftSideWidget)
        self.splitter.addWidget(self.plotWidget)

        Act_exit = QtWidgets.QAction(QtGui.QIcon('exit.png'), 'B&eenden', self)
        Act_exit.setShortcut('Ctrl+E')
        Act_exit.setStatusTip('Beenden')
        Act_exit.triggered.connect(QtWidgets.qApp.quit)
        menubar = self.menuBar()
        Menu_file = menubar.addMenu('&Datei')
        Menu_file.addAction(Act_exit)

        Act_help = QtWidgets.QAction(QtGui.QIcon('help.png'), '&Hilfe', self)
        Act_help.setShortcut('Ctrl+H')
        Act_help.setStatusTip('Hilfe anzeigen')
        Act_help.triggered.connect(self.show_help_message)
        Menu_help = menubar.addMenu('&Hilfe')
        Menu_help.addAction(Act_help)

    def show_help_message(self):
        help_text = (
            'Programm zur Erzeugung von Gleichzeitigkeitseffekten in '
            'Zeitreihen. \n\n'
            'Die Standardabweichung sigma steuert die Streuung der '
            'zeitlichen Verschiebung der Kopien. "Seed" kontrolliert '
            'den Ausgangspunkt des Zufallsgenerators und ermöglicht '
            'reproduzierbare Zufallsziehungen. \n'
            'Mit der Schaltfläche "Zeitverschiebung" kann eine '
            'Exceldatei geladen werden. Diese muss als erste Spalte '
            'Zeitstempel enthalten und in allen weiteren Spalten '
            'die zu kopierenden Zeitreihen. Die erste Zeile '
            'muss die Überschriften enthalten. Die Ausgabe erfolgt im selben '
            'Ordner als neue Exceldatei.'
            )

        QtWidgets.QMessageBox.information(self, 'Hilfe',
                                          help_text,
                                          QtWidgets.QMessageBox.Ok)


class MyMplCanvas(FigureCanvasQTAgg):
    """Canvas to draw Matplotlib figures on"""

    def __init__(self, width=5, height=4, dpi=100):
        # Define style settings for the plots
        try:  # Try to load personalized matplotlib style file
            if getattr(sys, 'frozen', False):  # If frozen with cx_Freeze
                homePath = os.path.dirname(sys.executable)
            else:  # Otherwise, if running unfrozen (e.g., within Spyder)
                homePath = os.path.dirname(__file__)
            mpl.style.use(os.path.join(homePath, './lpagg.mplstyle'))
        except OSError:
            pass

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(111)

    def update_figure(self, sigma, copies, seed):
        np.random.seed(seed)  # Fixing the seed makes the results persistent
        # Create a list of random values for all copies of the current column
        randoms = np.random.normal(0, sigma, copies)  # Array of random values
        randoms_int = [int(value) for value in np.round(randoms, 0)]
        limit = max(-1*min(randoms_int), max(randoms_int))
        bins = range(-limit, limit+2)

        mu = np.mean(randoms_int)
        sigma = np.std(randoms_int, ddof=1)
        title_mu_std = r'$\mu={:0.3f},\ \sigma={:0.3f}$'.format(mu, sigma)

        self.axes.cla()  # Clear axis
        self.axes.hist(randoms_int, bins, align='left', rwidth=0.9)
        self.axes.set_xlabel('Zeitschritte')
        self.axes.set_ylabel('Häufigkeit')
        self.axes.set_title(title_mu_std)
        self.axes.yaxis.grid(True)  # Activate grid on horizontal axis
        self.draw()


class SettingsGUI(QtWidgets.QWidget):
    """By inheriting from QtWidgets.QWidget our MainClassAsGUI becomes
    a QWidget, thus it gets e.g. the functions self.setLayout() and self.show()

    """
    def __init__(self, plotWidget, statusBar):
        super().__init__()  # Exec parent's init function

        self.statusBar = statusBar
        self.settings = dict(sigma=1, copies=10, seed=4)
        self.set_hist = dict(PNG=True, PDF=False)
        self.file_in = None
        self.file_out = None

        self.mainLayout = QtWidgets.QGridLayout()
        self.setLayout(self.mainLayout)

        self.tabsWidget = QtWidgets.QTabWidget()
        self.tab1Widget = QtWidgets.QWidget()
        self.tab1Layout = QtWidgets.QGridLayout()
        self.tab1Widget.setLayout(self.tab1Layout)
        self.tab2Widget = QtWidgets.QWidget()
        self.tab2Layout = QtWidgets.QGridLayout()
        self.tab2Widget.setLayout(self.tab2Layout)

        self.tabsWidget.addTab(self.tab1Widget, 'Eingabe')
        self.tabsWidget.addTab(self.tab2Widget, 'Ausgabe')
        self.tab1UI()
        self.tab2UI()

        self.button = QtWidgets.QPushButton('Zeitverschiebung')
        self.button.clicked.connect(self.perform_time_shift)

        self.mainLayout.addWidget(self.tabsWidget, 0, 0)
        self.mainLayout.addWidget(self.button, 1, 0,)

        self.plotWidget = plotWidget

        self.callback_lineEdits()  # Draw the first histogram

    def tab1UI(self):
        for i in range(self.tab1Layout.count()):
            self.tab1Layout.itemAt(i).widget().close()

        n = 0
        self.lineEdit_list = []
        for key, value in self.settings.items():
            lineEdit = QtWidgets.QLineEdit()
            lineEdit.setText(str(value))
            lineEdit.editingFinished.connect(self.callback_lineEdits)
            self.tab1Layout.addWidget(lineEdit, n, 1)
            self.tab1Layout.addWidget(QtWidgets.QLabel(key), n, 0)
            self.lineEdit_list.append(lineEdit)
            n += 1

    def tab2UI(self):
        n = 0
        label = QtWidgets.QLabel('Die Ausgabe erfolgt als Excel Datei, mit '
                                 'den Einstellungen für "copies" und "sigma" '
                                 'im Dateinamen: '
                                 'Beispieldatei_c10_s5.xlsx. '
                                 'Darüber hinaus können die '
                                 'Histogramme der Zufallsziehungen '
                                 'als .png oder .pdf abgespeichert werden.')
        label.setWordWrap(True)
        self.tab2Layout.addWidget(label, n, 0)
        n += 1
        self.cb_list = []
        for key, value in self.set_hist.items():
            cb = QtWidgets.QCheckBox(key+' Histogramm')
            cb.setChecked(value)
            cb.stateChanged.connect(self.callback_cbs)
            self.tab2Layout.addWidget(cb, n, 0)
            self.cb_list.append(cb)
            n += 1

    def callback_lineEdits(self):
        for i, key in enumerate(self.settings):
            try:
                self.settings[key] = float(self.lineEdit_list[i].text())
            except Exception:
                pass

        self.plotWidget.update_figure(self.settings['sigma'],
                                      int(self.settings['copies']),
                                      int(self.settings['seed']))

    def callback_cbs(self, state):
        for i, key in enumerate(self.set_hist):
            self.set_hist[key] = self.cb_list[i].isChecked()

    def perform_time_shift(self):
        load_dir = './'
        if getattr(sys, 'frozen', False):  # If frozen with cx_Freeze
            load_dir = os.path.expanduser('~user')

        self.file = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Bitte Exceldatei mit Zeitreihe auswählen',
                load_dir, 'Excel Datei (*.xlsx)')[0]

        if self.file == '' or self.file is None:
            return

        self.statusBar().showMessage('Bitte warten...')
        Qt.QApplication.setOverrideCursor(Qt.QCursor(Qt.Qt.WaitCursor))

        try:
            # Perform the time shift with the given settings
            res = lpagg.simultaneity.run(self.settings['sigma'],
                                                 int(self.settings['copies']),
                                                 int(self.settings['seed']),
                                                 self.file,
                                                 self.set_hist)
            output_file = res[0]
            GLF = res[1]

        except Exception as e:
            Qt.QApplication.restoreOverrideCursor()
            self.statusBar().showMessage('')
            QtWidgets.QMessageBox.critical(self, 'Fehler', str(e),
                                           QtWidgets.QMessageBox.Ok)
        else:
            Qt.QApplication.restoreOverrideCursor()
            self.statusBar().showMessage('GLF = {:0.1f}%'.format(GLF*100))
            # Show a MessageBox.
            QtWidgets.QMessageBox.information(self, 'Ausgabe erzeugt',
                                              output_file,
                                              QtWidgets.QMessageBox.Ok)


def main():
    import ctypes
    myappid = 'appid.lpagg.GUI'  # Define a unique AppID
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    app = QtWidgets.QApplication([])
    app.setAttribute(Qt.Qt.AA_EnableHighDpiScaling)
    gui = MainClassAsGUI()  # gui has inherited from QMainWindow
    gui.setWindowTitle('Gleichzeitigkeit')

    if getattr(sys, 'frozen', False):  # If frozen with cx_Freeze
        homePath = os.path.dirname(sys.executable)
        iconFile = os.path.join(homePath, 'icon.png')
    else:  # Otherwise, if running unfrozen (e.g., within Spyder)
        homePath = os.path.dirname(__file__)
        iconFile = os.path.join(homePath, '../res/icon.ico')

    app.setWindowIcon(QtGui.QIcon(iconFile))
    gui.setWindowIcon(QtGui.QIcon(iconFile))

    gui.show()
    app.exec_()


if __name__ == "__main__":
    main()
