#include <QApplication>
#include "tomowindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    TomoWindow w;

    w.show();

    return a.exec();
}
