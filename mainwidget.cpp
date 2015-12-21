#include "mainwidget.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QDir>

#include <valarray>

#include "testwidget.h"

MainWidget::MainWidget(QWidget *parent)
    : QWidget(parent)
{
    TestWidget *testWidget = new TestWidget(this);

    QPushButton *button1 = new QPushButton("test", this);

    QVBoxLayout *layout1 = new QVBoxLayout();
    QHBoxLayout *layout2 = new QHBoxLayout();

    layout1->addWidget(testWidget);
    layout1->addLayout(layout2);
    layout2->addStretch(1);
    layout2->addWidget(button1);

    connect(button1, SIGNAL(clicked()), testWidget, SLOT(start()));

    setLayout(layout1);
}

MainWidget::~MainWidget() {}

QSize MainWidget::sizeHint() const { return QSize(512, 512); }
