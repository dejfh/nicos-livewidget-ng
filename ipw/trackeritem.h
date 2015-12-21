#ifndef HPW_GRAPHICSTRACKERITEM_H
#define HPW_GRAPHICSTRACKERITEM_H

#include <QPen>
#include <QColor>
#include <QBrush>
#include <QGraphicsObject>
#include <QGraphicsLineItem>
#include <QGraphicsSimpleTextItem>

#include <functional>

#include "ipw/plot2dtransform.h"

namespace ipw
{

class TrackerItem : public QGraphicsObject
{
	Q_OBJECT

public:
	enum SelectionMode { DisableSelection, InvisibleSelection, LineSelection, RectSelection };

	TrackerItem(Plot2DTransform *transform, const QPen &crossPen = QPen(QColor(0, 0, 0, 192)), const QBrush &textBrush = QBrush(Qt::black),
						const QBrush &textBackgroundBrush = QBrush(QColor(255, 255, 255, 128)), const QPen &selectionPen = QPen(QColor(255, 255, 0)), QGraphicsItem *parent = 0);
	~TrackerItem();

	virtual QRectF boundingRect() const;
	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	Plot2DTransform *transform() const;
	void setTransform(Plot2DTransform *transform);

	std::function<QString(QPointF)> hoverLookup() const;
	void setHoverLookup(std::function<QString(QPointF)> hoverLookup);

	SelectionMode selectionMode() const;
	void setSelectionMode(SelectionMode selectionMode);

protected:
	virtual void hoverMoveEvent(QGraphicsSceneHoverEvent *event);
	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

	virtual QVariant itemChange(GraphicsItemChange change, const QVariant &value);

private slots:
	void updateText();

private:
	Plot2DTransform *m_transform;
	std::function<QString(QPointF)> m_hoverLookup;
	SelectionMode m_selectionMode;

	QGraphicsLineItem *m_horizontalLine;
	QGraphicsLineItem *m_verticalLine;
	QGraphicsSimpleTextItem *m_textItem;
	QGraphicsRectItem *m_textBackgroundItem;
	QGraphicsLineItem *m_selectLineItem;
	QGraphicsRectItem *m_selectRectItem;

	bool m_selecting;
	QPointF m_selectStart;
	bool m_panning;
	QPointF m_panStart;

signals:
	void selectionComplete(QRectF selection);
	void rightClick(QPointF point);
};

} // namespace hpw

#endif // HPW_GRAPHICSTRACKERITEM_H
