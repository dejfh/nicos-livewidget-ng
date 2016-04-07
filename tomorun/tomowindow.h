#ifndef TOMOWINDOW_H
#define TOMOWINDOW_H

#include <QMainWindow>

namespace Ui {
class TomoWindow;
}

class TomoWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit TomoWindow(QWidget *parent = 0);
    ~TomoWindow();

private:
    Ui::TomoWindow *ui;
};

#endif // TOMOWINDOW_H
