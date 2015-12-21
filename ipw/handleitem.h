#ifndef HPW_GRAPHICSGRIPITEM_H
#define HPW_GRAPHICSGRIPITEM_H

#include <QGraphicsObject>
#include <QGraphicsRectItem>

#include "ipw/plot2dtransform.h"

namespace ipw
{

/**
 * @brief The HandleItem class provides a handle for a ImagePlot, that can be dragged by the user.
 */
class HandleItem : public QGraphicsObject
{
	Q_OBJECT

public:
	HandleItem(Plot2DTransform *transform, QGraphicsItem *parent = 0);
	~HandleItem();

	virtual QRectF boundingRect() const;
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	QRectF rect() const;
	void setRect(QRectF rect);

signals:
	void rightClicked(QPoint screenPos);
	void dragStarted(QPointF dataPoint);
	void dragContinued(QPointF dataPoint);
	void dragDone(QPointF dataPoint);

protected:
	virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);
	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

private:
	Plot2DTransform *m_transform;
	QGraphicsRectItem *m_rectItemBack;
	QGraphicsRectItem *m_rectItemFront;
	QPointF m_dragSceneOffset;
};

} // namespace hpw

#endif // HPW_GRAPHICSGRIPITEM_H
