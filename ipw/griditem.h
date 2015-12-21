#ifndef GRAPHICSGRIDITEM_H
#define GRAPHICSGRIDITEM_H

#include <QPen>
#include <QPointF>
#include <QRectF>
#include <QGraphicsObject>

#include "ipw/plot2dtransform.h"

namespace ipw
{

/**
 * @brief The GridItem class provides a grid for an ImagePlotWidget.
 */
class GridItem : public QGraphicsObject
{
	Q_OBJECT

public:
	GridItem(Plot2DTransform *transform, int minDistance = 180, const QPen &pen = QPen(QColor(0, 0, 255, 128), 1, Qt::SolidLine, Qt::FlatCap, Qt::MiterJoin), QGraphicsItem *parent = 0);
	~GridItem();

	Plot2DTransform *transform() const;

	int minDistance() const;
	void setMinDistance(int distance);

	bool useManualInterval() const;
	void setUseManualInterval(bool use);

	QSizeF manualInterval() const;
	void setManualInterval(double interval);
	void setManualInterval(QSizeF interval);

	const QPen &pen() const;
	void setPen(const QPen &pen);

	virtual QRectF boundingRect() const;
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	QRectF firstDataInterval(QRectF dataRect) const;
	QRectF firstSceneInterval(QRectF sceneRect) const;

private slots:
	void updateToScale();

private:
	Plot2DTransform *m_transform;
	int m_minDistance;
	QPen m_pen;
	QSizeF m_manualInterval;
	bool m_useManualInterval;
};

} // namespace hpw

#endif // GRAPHICSGRIDITEM_H
