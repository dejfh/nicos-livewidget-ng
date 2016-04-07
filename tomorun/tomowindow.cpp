#include "tomowindow.h"
#include "ui_tomowindow.h"

TomoWindow::TomoWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::TomoWindow)
{
    ui->setupUi(this);
}

TomoWindow::~TomoWindow()
{
    delete ui;
}
