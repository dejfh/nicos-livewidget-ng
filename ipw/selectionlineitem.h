#ifndef HPW_SELECTIONLINEITEM_H
#define HPW_SELECTIONLINEITEM_H

#include <QGraphicsObject>
#include <QGraphicsLineItem>
#include <QRectF>
#include <QPoint>

#include "ipw/plot2dtransform.h"
#include "ipw/handleitem.h"

#include <functional>

namespace ipw
{

class SelectionLineItem : public QGraphicsObject
{
	Q_OBJECT

public:
	SelectionLineItem(Plot2DTransform *transform, QGraphicsItem *parent = 0);
	~SelectionLineItem();

	virtual QRectF boundingRect() const;
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	QLineF line() const;
	void setLine(QLineF line);

signals:
	void selectionChanged(QLineF line);
	void rightClicked(QLineF selection, QPoint point);

private slots:
	void updateGeometry();
	void gripRightClick(QPoint screenPoint);
	void gripDragStart(QPointF dataPoint);
	void gripDragContinue(QPointF dataPoint);
	void gripDragEnd(QPointF dataPoint);

private:
	enum Grips { StartGrip = 0, EndGrip = 1, MidGrip = 2, GripCount = 3, NoGrip = -1 };

	QLineF m_line;

	Plot2DTransform *m_transform;
	QGraphicsLineItem *m_lineItemBack;
	QGraphicsLineItem *m_lineItemFront;
	HandleItem *m_gripItems[GripCount];

	int m_dragIndex;
};

} // namespace hpw

#endif // HPW_SELECTIONLINEITEM_H
