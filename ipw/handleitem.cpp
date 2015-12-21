#include "handleitem.h"

#include <QBrush>
#include <QPen>

#include <QGraphicsSceneMouseEvent>

#include "ipw/geometryhelper.h"

namespace ipw
{

HandleItem::HandleItem(Plot2DTransform *transform, QGraphicsItem *parent)
	: QGraphicsObject(parent)
	, m_transform(transform)
	, m_rectItemBack(new QGraphicsRectItem(this))
	, m_rectItemFront(new QGraphicsRectItem(this))
{
	setFlag(ItemHasNoContents);
	setAcceptHoverEvents(true);
	setAcceptTouchEvents(true);
	setAcceptedMouseButtons(Qt::LeftButton | Qt::RightButton);

	m_rectItemBack->setVisible(false);
	m_rectItemBack->setBrush(QColor(0, 0, 0, 32));
	m_rectItemBack->setPen(Qt::NoPen);
	m_rectItemFront->setVisible(false);
	m_rectItemFront->setPen(QPen(QColor(255, 255, 255, 128), 1, Qt::SolidLine, Qt::FlatCap, Qt::MiterJoin));

	QRectF rect(-4, -4, 8, 8);
	m_rectItemFront->setRect(rect);
	double adjust = m_rectItemFront->pen().widthF() * 1.5;
	rect.adjust(-adjust, -adjust, adjust, adjust);
	m_rectItemBack->setRect(rect);
}

HandleItem::~HandleItem()
{
}

QRectF HandleItem::boundingRect() const
{
	return rect();
}

void HandleItem::paint(QPainter *, const QStyleOptionGraphicsItem *, QWidget *)
{
}

QRectF HandleItem::rect() const
{
	return m_rectItemBack->rect();
}

void HandleItem::setRect(QRectF rect)
{
	prepareGeometryChange();
	m_rectItemFront->setRect(rect);
	double adjust = m_rectItemFront->pen().widthF() * 1.5;
	rect.adjust(-adjust, -adjust, adjust, adjust);
	m_rectItemBack->setRect(rect);
}

void HandleItem::hoverMoveEvent(QGraphicsSceneHoverEvent *)
{
}

void HandleItem::hoverEnterEvent(QGraphicsSceneHoverEvent *)
{
	m_rectItemBack->setVisible(true);
	m_rectItemFront->setVisible(true);
}

void HandleItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *)
{
	m_rectItemBack->setVisible(false);
	m_rectItemFront->setVisible(false);
}

void HandleItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	emit dragContinued(m_transform->sceneToData(event->scenePos() - m_dragSceneOffset));
}

void HandleItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	switch (event->button()) {
	case Qt::LeftButton:
		event->accept();
		setZValue(1);
		m_dragSceneOffset = event->scenePos() - this->scenePos();
		emit dragStarted(m_transform->sceneToData(this->scenePos()));
		return;
	case Qt::RightButton:
		event->accept();
		emit rightClicked(event->screenPos());
		break;
	default:
		event->ignore();
		return;
	}
}

void HandleItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	switch (event->button()) {
	case Qt::LeftButton:
		event->accept();
		setZValue(0);
		emit dragDone(m_transform->sceneToData(event->scenePos() - m_dragSceneOffset));
		return;
	case Qt::RightButton:
		event->accept();
		return;
	default:
		event->ignore();
		return;
	}
}

} // namespace hpw
