#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "ipw/imageplot.h"

#include "helper/helper.h"

using hlp::assert_true;

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
{
	ui->setupUi(this);

}

MainWindow::~MainWindow()
{
	delete ui;
}
