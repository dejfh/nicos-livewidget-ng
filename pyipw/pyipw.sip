
%Module(name=nicosimageplot, version=1)

%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

class ImagePlot : QWidget
{
%TypeHeaderCode
#include "pyipw.h"
%End
private:
    ImagePlot(const ImagePlot& other);

public:
    ImagePlot(QWidget *parent = 0);

    const QPixmap pixmap() const;
    void setPixmap(const QPixmap &pixmap);
    void setImage(const QImage &image);

//    SelectionMode selectionMode() const;
//    void setSelectionMode(SelectionMode mode);

//    GridMode gridMode() const;
//    void setGridMode(GridMode mode);
    QSizeF gridInterval() const;
    void setGridInterval(double interval);

    QRectF selectionRect() const;
    QLineF selectionLine() const;

    void setSelection(QRectF rect);
    void setSelection(QLineF line);
    void setSelection(QPointF start, QPointF end);
    void setSelection(QPointF start, QSizeF size);
    void resetSelection();

    QGraphicsView *graphicsView() const;

public slots:
    void autoScale();
    void setView(QRectF dataRect);
    void setView(QPointF end, QPointF start);
};

/*
class RangeSelectWidget
{
%TypeHeaderCode
#include "pyipw.h"
%End

};

class HistogramPlot
{
%TypeHeaderCode
#include "pyipw.h"
%End

};
*/

/*
enum SelectionMode {
    SelectionDisable,
    SelectionZoom,
    SelectionRect,
    SelectionLine
};
enum GridMode {
    GridDisable,
    GridAuto,
    GridCustom
};
*/

