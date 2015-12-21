#ifndef PLOT2DTRANSFORM_H
#define PLOT2DTRANSFORM_H

#include <QObject>
#include <QPointF>
#include <QRectF>
#include <QSizeF>

namespace ipw {

/**
 * @brief The Plot2DTransform interface transforms points between data space and screen space.
 */
class Plot2DTransform : public QObject {
    Q_OBJECT

protected:
    Plot2DTransform(QObject *parent=0) : QObject(parent) {}
    virtual ~Plot2DTransform() {}

public:
    virtual QPointF origin() const = 0;
    virtual void setOrigin(QPointF scenePoint) = 0;
    virtual void pan(QPointF dataPoint, QPointF scenePoint) = 0;

    virtual QSizeF sceneToData(QSizeF sceneSize) const = 0;
    virtual QSizeF dataToScene(QSizeF dataSize) const = 0;

    virtual QPointF sceneToData(QPointF scenePoint) const = 0;
    virtual QPointF dataToScene(QPointF dataPoint) const = 0;

    virtual QRectF sceneToData(QRectF sceneRect) const = 0;
    virtual QRectF dataToScene(QRectF dataRect) const = 0;

signals:
    void viewChanged();
};

} // namespace hpw

#endif // PLOT2DTRANSFORM_H
