#ifndef HPW_GRAPHICSSELECTIONITEM_H
#define HPW_GRAPHICSSELECTIONITEM_H

#include <QGraphicsObject>
#include <QGraphicsRectItem>
#include <QRectF>
#include <QPoint>

#include "ipw/plot2dtransform.h"
#include "ipw/handleitem.h"

#include <functional>

namespace ipw
{

class SelectionRectItem : public QGraphicsObject
{
	Q_OBJECT

public:
	SelectionRectItem(Plot2DTransform *transform, QGraphicsItem *parent = 0);
	~SelectionRectItem();

	virtual QRectF boundingRect() const;
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	QRectF rect() const;
	void setRect(QRectF rect);

signals:
	void selectionChanged(QRectF selection);
	void rightClicked(QRectF selection, QPoint point);

private slots:
	void updateGeometry();
	void gripRightClick(QPoint screenPoint);
	void gripDragStart(QPointF dataPoint);
	void gripDragContinue(QPointF dataPoint);
	void gripDragEnd(QPointF dataPoint);

private:
	enum Grips {
		TopLeftGrip = 0,
		TopGrip = 1,
		TopRightGrip = 2,
		LeftGrip = 3,
		CenterGrip = 4,
		RightGrip = 5,
		BottomLeftGrip = 6,
		BottomGrip = 7,
		BottomRightGrip = 8,
		GripCount = 9,
		NoGrip = -1
	};

	QRectF m_rect;

	Plot2DTransform *m_transform;
	QGraphicsRectItem *m_rectItemBack;
	QGraphicsRectItem *m_rectItemFront;
	HandleItem *m_gripItems[GripCount];

	int m_dragIndex;
};

} // namespace hpw

#endif // HPW_GRAPHICSSELECTIONITEM_H
