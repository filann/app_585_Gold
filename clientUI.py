import os
import csv
import json
import requests
import base64
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *


# to-do
#
# custom tables
# choose target

class TweakerUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.dataJSON = None
        self.uuid = None


        self.configureGeometry()
        self.setWindowTitle('Tweaker')
        self.setCentralWidget(QStackedWidget())

        dataPanel = QWidget()
        self.centralWidget().addWidget(dataPanel)
        dataPanelLayout = QGridLayout()
        dataPanel.setLayout(dataPanelLayout)
        self.configureDataPanel(dataPanelLayout)

        featPanel = QWidget()
        self.centralWidget().addWidget(featPanel)
        featPanelLayout = QGridLayout()
        featPanel.setLayout(featPanelLayout)
        self.configureFeatPanel(featPanelLayout)

        treePanel = QWidget()
        self.centralWidget().addWidget(treePanel)
        treePanelLayout = QGridLayout()
        treePanel.setLayout(treePanelLayout)
        self.configureTreePanel(treePanelLayout)

    def configureGeometry(self):
        screenGeometry = QDesktopWidget().availableGeometry()
        windowGeometry = QStyle.alignedRect(Qt.LeftToRight,
                                            Qt.AlignCenter,
                                            screenGeometry.size() * 0.6,
                                            screenGeometry)
        self.setGeometry(windowGeometry)

    def configureDataPanel(self, panelLayout):
        importDataButton = QPushButton('Import CSV')
        panelLayout.addWidget(importDataButton)
        importDataButton.clicked.connect(self.importData)
        importDataButton.setShortcut(QKeySequence('Ctrl+O'))
        importDataButton.setSizePolicy(*[QSizePolicy.Minimum] * 2)

        self.dataTable = QTableWidget()
        panelLayout.addWidget(self.dataTable, 1, 0, 1, 2)
        self.dataTable.setEnabled(False)

        self.modeSelectionBox = QFrame()
        panelLayout.addWidget(self.modeSelectionBox, 0, 1)
        boxLayout = QGridLayout()
        self.modeSelectionBox.setLayout(boxLayout)
        self.modeSelectionBox.setEnabled(False)

        classificationMode = QRadioButton('Classification')
        boxLayout.addWidget(classificationMode, 1, 0)

        regressionMode = QRadioButton('Regression')
        boxLayout.addWidget(regressionMode, 1, 1)

        decisionMode = QRadioButton('Decision Tree')
        boxLayout.addWidget(decisionMode, 1, 2)

        self.modeSelector = QButtonGroup(self)
        self.modeSelector.addButton(classificationMode, id=1)
        self.modeSelector.addButton(regressionMode, id=0)
        self.modeSelector.addButton(decisionMode, id=2)
        self.modeSelector.buttonToggled.connect(self.readyState)

        self.processDataButton = QPushButton('Process data')
        boxLayout.addWidget(self.processDataButton, 0, 0, 1, 2)
        self.processDataButton.clicked.connect(self.processData)
        self.processDataButton.setEnabled(False)

    def configureFeatPanel(self, panelLayout):
        self.showNTopFeatControl = QSpinBox()
        panelLayout.addWidget(self.showNTopFeatControl)
        self.showNTopFeatControl.valueChanged.connect(self.showNTopFeat)
        self.showNTopFeatControl.setPrefix('Show top ')
        self.showNTopFeatControl.setSuffix(' features')

        showAllFeatButton = QPushButton('Show all features')
        panelLayout.addWidget(showAllFeatButton)
        showAllFeatButton.clicked.connect(self.showAllFeat)

        separateGroupingButton = QPushButton('Separate grouping')
        panelLayout.addWidget(separateGroupingButton, 0, 1)
        separateGroupingButton.clicked.connect(self.separateGrouping)

        resetGroupingButton = QPushButton('Reset grouping')
        panelLayout.addWidget(resetGroupingButton, 1, 1)
        resetGroupingButton.clicked.connect(self.resetGrouping)

        toDataPanelButton = QPushButton('Switch to data panel')
        panelLayout.addWidget(toDataPanelButton, 0, 2)
        toDataPanelButton.clicked.connect(self.toDataPanel)

        plotButton = QPushButton('Plot')
        panelLayout.addWidget(plotButton, 1, 2)
        plotButton.clicked.connect(self.plot)

        self.featTable = QTableWidget()
        self.featTable.setColumnCount(2)
        panelLayout.addWidget(self.featTable, 2, 0, 1, 3)


    def configureTreePanel(self, panelLayout):
        showDifYearButton = QPushButton('Difference between trees of each year')
        panelLayout.addWidget(showDifYearButton, 0, 0)
        showDifYearButton.clicked.connect(self.showDifYear)
        showDifYearButton.setFixedSize(266, 30)

        showImFeatButton = QPushButton('Show important features')
        panelLayout.addWidget(showImFeatButton, 0, 1)
        showImFeatButton.clicked.connect(self.showImFeat)
        showImFeatButton.setFixedSize(266, 30)

        showPathSampleButton = QPushButton('Show path for sample')
        panelLayout.addWidget(showPathSampleButton, 0, 2)
        showPathSampleButton.clicked.connect(self.showPathSample)
        showPathSampleButton.setFixedSize(266, 30)

        toDataPanelButton = QPushButton('Switch to data panel')
        panelLayout.addWidget(toDataPanelButton, 2, 2)
        toDataPanelButton.clicked.connect(self.toDataPanel)

        self.picturePanel = QTabWidget()
        panelLayout.addWidget(self.picturePanel, 1, 0, 1, 3)
        self.picturePanel.setEnabled(False)

        self.textPanel = QLabel()
        # self.textPanel.setText()
        panelLayout.addWidget(self.textPanel, 2, 0, 1, 2)

    def importData(self):
        csvPath = QFileDialog.getOpenFileName(caption='Open CSV file',
                                              directory='~',
                                              filter='CSV files (*.csv)')[0]
        if csvPath:
            with open(csvPath, newline='') as csvFile:
                csvDialect = csv.Sniffer().sniff(csvFile.read(1024))
                csvFile.seek(0)
                dataCSV = csv.reader(csvFile, csvDialect)
                dataHeader = next(dataCSV)

                self.dataJSON = dict(zip(dataHeader, [dict() for _ in dataHeader]))

                self.dataTable.setColumnCount(len(dataHeader))
                self.dataTable.setHorizontalHeaderLabels(dataHeader)
                for i, row in enumerate(dataCSV):
                    self.dataTable.insertRow(i)
                    for j, value in enumerate(row):
                        self.dataJSON[dataHeader[j]][str(i)] = value
                        self.dataTable.setItem(i, j, QTableWidgetItem(value))
            self.dataTable.setEnabled(True)
            self.dataTable.setSortingEnabled(True)
            self.modeSelectionBox.setEnabled(True)

    def processData(self):
        try:
            requestJSON = {'mode': self.selectedMode,
                               'data': self.dataJSON,
                               'target': 'Y'}
            response = requests.post(
                    url='http://127.0.0.1:8000/tweaker/v1/uploads',
                    json=json.dumps(requestJSON),
                    headers={'Content-type': 'application/json',
                             'Accept': 'application/json'})
            if response.status_code is 201:
                if self.selectedMode == 'D':
                    # self.url = response.json()['url']
                    self.centralWidget().setCurrentIndex(2)
                else:
                    self.uuid = response.json()['uuid']
                    featRanking = response.json()['featRanking']
                    self.featTable.setRowCount(len(featRanking))
                    for i, featNameAndRank in enumerate(featRanking):
                        featName = QTableWidgetItem(featNameAndRank[0])
                        featRank = QTableWidgetItem(str(featNameAndRank[1]))
                        self.featTable.setItem(i, 0, featName)
                        self.featTable.setItem(i, 1, featRank)
                    self.showNTopFeatControl.setMaximum(self.featTable.rowCount())
                    self.centralWidget().setCurrentIndex(1)
            else:
                # let user know
                pass
        except requests.exceptions.ConnectionError:
            pass


    def readyState(self):
        self.processDataButton.setEnabled(self.selectedMode is not None)

    def showNTopFeat(self):
        pass

    def showAllFeat(self):
        pass

    def resetGrouping(self):
        pass

    def separateGrouping(self):
        pass

    def toDataPanel(self):
        self.centralWidget().setCurrentIndex(0)

    def plot(self):
        pass

    def showImFeat(self):
        try:
            requestJSON = {'mode': self.selectedMode,
                           'data': self.dataJSON,
                           'target': 'Y'}

            response = requests.post(
                url='http://127.0.0.1:8000/tweaker/v1/uploads',
                json=json.dumps(requestJSON),
                headers={'Content-type': 'application/json',
                         'Accept': 'application/json'})
            if response.status_code is 201:
                self.picturePanel.clear()
                for p, n, t in zip(response.json()['pics'], response.json()['names'], response.json()['texts']):
                    scroll = QScrollArea()
                    image = QPixmap(p)

                    label = QLabel()
                    label.setPixmap(image)

                    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                    scroll.setWidget(label)

                    # название вкладки совпадает название класса
                    self.picturePanel.addTab(scroll, n)
                    self.textPanel.setText(t)


                self.picturePanel.setEnabled(True)
                self.centralWidget().setCurrentIndex(2)
            else:
                # let user know
                pass
        except requests.exceptions.ConnectionError:
            pass

    def showPathSample(self):
        pass

    def showDifYear(self):
        try:
            requestJSON = {'mode': self.selectedMode,
                           'data': self.dataJSON,
                           'target': 'Y'}

            response = requests.post(
                url='http://127.0.0.1:8000/tweaker/v1/uploads',
                json=json.dumps(requestJSON),
                headers={'Content-type': 'application/json',
                         'Accept': 'application/json'})
            if response.status_code is 201:
                self.picturePanel.clear()
                for tree in response.json()['trees']:
                    scroll = QScrollArea()
                    image = QPixmap(tree)

                    label = QLabel()
                    label.setPixmap(image)

                    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                    scroll.setWidget(label)

                    # название вкладки совпадает название класса
                    self.picturePanel.addTab(scroll, tree[-13:-4])

                self.picturePanel.setEnabled(True)
                self.centralWidget().setCurrentIndex(2)
            else:
                # let user know
                pass
        except requests.exceptions.ConnectionError:
            pass

    @property
    def selectedMode(self):
        return ['C', 'R','D', None][self.modeSelector.checkedId()]


if __name__ == '__main__':
    import sys

app = QApplication(sys.argv)
tweaker = TweakerUI()
tweaker.show()
sys.exit(app.exec_())
