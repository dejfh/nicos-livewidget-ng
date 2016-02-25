#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "ipw/imageplot.h"

#include "helper/helper.h"

using hlp::assert_true;

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
	, filterChain(new FilterChain(this))
{
	ui->setupUi(this);

	filterChain->setInputFitsFile("C:/Dev/huge data/Tomography/lava/raw/seismolava64__000.000.fits");
	filterChain->setDarkImageFitsFile("C:/Dev/huge data/Tomography/lava/darkimage/di_seismolava64__1.fits");
	filterChain->setOpenBeamFitsFile("C:/Dev/huge data/Tomography/lava/openbeam/ob_seismolava64__1.fits");

	ipw::ImagePlot *imagePlotWidget = new ipw::ImagePlot(this);
	this->setCentralWidget(imagePlotWidget);

	assert_true() << connect(filterChain, SIGNAL(pixmapChanged(QImage)), imagePlotWidget, SLOT(setImage(QImage)));

	filterChain->start();
}

MainWindow::~MainWindow()
{
	delete ui;
}
