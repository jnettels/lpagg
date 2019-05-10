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
import logging
import ctypes
import traceback
import matplotlib as mpl
import lpagg.simultaneity
from PyQt5 import QtWidgets, Qt, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Define the logging function
logger = logging.getLogger(__name__)


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
    '''Canvas to draw Matplotlib figures on.
    Is used for both the embedded histogram plot and the line plot.
    '''

    def __init__(self, width=5, height=4, dpi=100):
        # Define style settings for the plots
        try:  # Try to load personalized matplotlib style file
            if getattr(sys, 'frozen', False):  # If frozen with cx_Freeze
                homePath = os.path.dirname(sys.executable)
            else:  # Otherwise, if running unfrozen (e.g., within Spyder)
                homePath = os.path.dirname(__file__)
            mpl.style.use(os.path.join(homePath, './lpagg.mplstyle'))
        except OSError as e:
            logger.debug(e)
            pass

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(111)

    def update_histogram(self, sigma, copies, seed):
        '''Create or update a histogram plot'''
        np.random.seed(seed)  # Fixing the seed makes the results persistent

        # Create a list of random values for all copies of the current column
        if copies == 0:  # No copies (instead original profile is shifted)
            randoms = np.random.normal(0, sigma, 1)
        else:  # Normal usage: Only copies are shifted
            randoms = np.random.normal(0, sigma, copies)
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

    def update_lineplot(self, df):
        '''Create or update a line plot with specific columns from the
        resulting DataFrame'''
        self.axes.cla()  # Clear axis
        self.axes.plot(df['Shift'], label='Shift')
        self.axes.plot(df['Reference'], '--', label='Referenz')
        self.axes.axhline(df['Shift'].max(), linestyle='-.',
                          label='max(Shift)', color='#e8d654')
        self.axes.axhline(df['Reference'].max(), linestyle='-.',
                          label='max(Referenz)', color='#5eccf3')
        self.fig.legend(loc='upper center', ncol=5)
        self.axes.yaxis.grid(True)  # Activate grid on horizontal axis
        self.draw()


class PlotWindow(QtWidgets.QWidget):
    '''New window for plot figures.
    '''
    def __init__(self):
        super().__init__()
        self.plot_layout = QtWidgets.QVBoxLayout(self)
        self.plot_canvas = MyMplCanvas(width=10, height=4, dpi=100)
        self.navi_toolbar = NavigationToolbar2QT(self.plot_canvas, self)
        self.plot_layout.addWidget(self.navi_toolbar)
        self.plot_layout.addWidget(self.plot_canvas)  # the matplotlib canvas

    def update_lineplot(self, df):
        self.setWindowTitle('Zeitreihen')
        self.plot_canvas.update_lineplot(df)


class SettingsGUI(QtWidgets.QWidget):
    '''The left part of the splitter in the MainWindow. All input boxes and
    buttons are created here. This part is devided into several tabs.

    '''
    def __init__(self, plotWidget, statusBar):
        super().__init__()  # Exec parent's init function

        self.statusBar = statusBar
        self.settings = dict(sigma=1, copies=10, seed=4)
        self.set_hist = dict(PNG=True, PDF=False)
        self.plot_timeseries = True
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

        self.start_button = QtWidgets.QPushButton('Zeitverschiebung')
        self.start_button.clicked.connect(self.perform_time_shift)

        self.mainLayout.addWidget(self.tabsWidget, 0, 0)
        self.mainLayout.addWidget(self.start_button, 1, 0,)

        self.plotWidget = plotWidget

        self.callback_lineEdits()  # Draw the first histogram

    def tab1UI(self):
        '''Tab with the main settings.
        '''
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
        '''Tab with other settings (for histogram plots)
        '''
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

        # Create checkboxes for histogram settings
        self.cb_list = []
        for key, value in self.set_hist.items():
            cb = QtWidgets.QCheckBox(key+' Histogramm speichern')
            cb.setChecked(value)
            cb.stateChanged.connect(self.callback_cbs)
            self.tab2Layout.addWidget(cb, n, 0)
            self.cb_list.append(cb)
            n += 1

        # Create checkbox for additional settings
        cb = QtWidgets.QCheckBox('Zeitreihe anzeigen')
        cb.setChecked(self.plot_timeseries)
        cb.stateChanged.connect(self.callback_cb_ts)
        self.tab2Layout.addWidget(cb, n, 0)
        n += 1

    def callback_lineEdits(self):
        '''This callback is called when any of the lineEdit widgets are
        edited. It updates the histogram with the new settings
        '''
        for i, key in enumerate(self.settings):
            settings_copy = self.settings[key]
            try:
                self.settings[key] = float(self.lineEdit_list[i].text())
                self.plotWidget.update_histogram(self.settings['sigma'],
                                                 int(self.settings['copies']),
                                                 int(self.settings['seed']))
            except Exception as e:
                self.settings[key] = settings_copy
                self.lineEdit_list[i].setText(str(settings_copy))
                logger.error(e)
                QtWidgets.QMessageBox.critical(self, 'Fehler', str(e),
                                               QtWidgets.QMessageBox.Ok)

    def callback_cbs(self, state):
        '''This callback is called when any of the checkbox widgets are
        changed. It stores the user input.
        '''
        for i, key in enumerate(self.set_hist):
            self.set_hist[key] = self.cb_list[i].isChecked()

    def callback_cb_ts(self, state):
        '''This callback is called when the timeseries checkbox widget is
        changed. It stores the user input.
        '''
        self.plot_timeseries = state

    def perform_time_shift(self):
        '''Callback for the button that starts the simultaneity calculation.
        To prevent the GUI from becoming unresponsive, an extra object for
        the task is created and moved to a thread of its own. Several
        connections for starting and finishing the tasks need to be made.
        '''
        load_dir = './'
#        if getattr(sys, 'frozen', False):  # If frozen with cx_Freeze
#            logger.debug(os.getcwd())
#            load_dir = os.path.expanduser('~user')

        self.file = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Bitte Exceldatei mit Zeitreihe auswählen',
                load_dir, 'Excel Datei (*.xlsx)')[0]

        if self.file == '' or self.file is None:
            return

        self.start_button.setEnabled(False)
        self.statusBar().showMessage('Bitte warten...')
        Qt.QApplication.setOverrideCursor(Qt.QCursor(Qt.Qt.WaitCursor))

        try:  # Thread magic to prevent unresponsive GUI
            self.objThread = QtCore.QThread()
            self.obj = self.simultaneity_obj(self.settings, self.file,
                                             self.set_hist)
            self.obj.moveToThread(self.objThread)
            self.objThread.started.connect(self.obj.run)
            self.obj.failed.connect(self.fail)
            self.obj.failed.connect(self.objThread.quit)
            self.obj.finished.connect(self.done)
            self.obj.finished.connect(self.objThread.quit)
            self.objThread.start()

        except Exception as e:
            logger.exception(e)
            Qt.QApplication.restoreOverrideCursor()
            self.statusBar().showMessage('')
            self.start_button.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, 'Fehler', str(e),
                                           QtWidgets.QMessageBox.Ok)
        else:
            logger.debug('Waiting for return of results...')
            pass

    def done(self, result):
        '''Function which is connected to the ``finished`` signal from
        ``simultaneity_obj``. Reads the returned result object and updates the
        status information.
        Also creates a line plot of the time series in a new window.
        '''
        output_file = result['output']
        GLF = result['GLF']
        Qt.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage('GLF = {:0.1f}%'.format(GLF*100))
        # Show a MessageBox.
        QtWidgets.QMessageBox.information(self, 'Ausgabe erzeugt',
                                          str(output_file),
                                          QtWidgets.QMessageBox.Ok)
        self.start_button.setEnabled(True)

        if self.plot_timeseries:
            try:  # Create a line plot of the time series in a new window
                self.plot_window = PlotWindow()
                self.plot_window.update_lineplot(result['df_sum'])
                self.plot_window.show()
            except Exception as e:
                logger.exception(e)
                QtWidgets.QMessageBox.critical(self, 'Fehler', str(e),
                                               QtWidgets.QMessageBox.Ok)

    def fail(self, error):
        '''Function which is connected to the ``failed`` signal from
        ``simultaneity_obj``. Is called if an exception occurs.
        '''
        Qt.QApplication.restoreOverrideCursor()

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle('Fehler')
        msg.setText('Bei der Verarbeitung der Zeitreihen ist ein Fehler '
                    'aufgetreten. Siehe die vollständige Fehlermeldung für '
                    'zusätzliche Informationen.')
        msg.setDetailedText(error)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.setEscapeButton(QtWidgets.QMessageBox.Ok)
        msg.exec_()

        self.start_button.setEnabled(True)
        self.statusBar().showMessage('')

    class simultaneity_obj(QtCore.QObject):
        '''We want to perform the simultaneity calculation, which can take
        several seconds, in a seperate thread. This prevents the GUI from
        locking up or becoming unresponsive. In order for this to work,
        the long running task is performed by a seperate object, which this
        class defines.
        '''
        finished = QtCore.pyqtSignal('PyQt_PyObject')  # for emitting results
        failed = QtCore.pyqtSignal('PyQt_PyObject')  # for emitting results

        def __init__(self, settings, file, set_hist):
            super().__init__()
            self.settings = settings
            self.file = file
            self.set_hist = set_hist

        def __del__(self):
            logger.debug('simultaneity object deleted')

        @QtCore.pyqtSlot()
        def run(self):
            # Perform the time shift with the given settings
            try:
                result = lpagg.simultaneity.run(self.settings['sigma'],
                                                int(self.settings['copies']),
                                                int(self.settings['seed']),
                                                self.file,
                                                self.set_hist)
            except Exception as e:
                logger.exception(e)
                error = traceback.format_exc()
                self.failed.emit(error)
            else:
                self.finished.emit(result)


def main():
    log_level = 'error'
    logger.setLevel(level=log_level.upper())  # Logger for this module
    logging.basicConfig(format='%(asctime)-15s %(message)s')
    logging.getLogger('lpagg.simultaneity').setLevel(level=log_level.upper())

    if sys.platform == 'win32':
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
