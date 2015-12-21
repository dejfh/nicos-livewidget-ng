#include "trackeritem.h"

#include <QGraphicsSceneMouseEvent>
#include <QCursor>

#include <cassert>

#include "ipw/geometryhelper.h"

namespace ipw
{

TrackerItem::TrackerItem(Plot2DTransform *transform, const QPen &crossPen, const QBrush &textBrush, const QBrush &textBackgroundBrush, const QPen &selectionPen, QGraphicsItem *parent)
	: QGraphicsObject(parent)
	, m_transform(transform)
	, m_selectionMode(RectSelection)
	, m_horizontalLine(0)
	, m_verticalLine(0)
	, m_textItem(0)
	, m_textBackgroundItem(0)
	, m_selectLineItem(0)
	, m_selectRectItem(0)
	, m_selecting(false)
	, m_panning(false)
{
	assert(transform && "There has to be a heatmap item to be controlled.");

	setFlag(ItemHasNoContents);
	setFlag(ItemSendsScenePositionChanges);
	setAcceptHoverEvents(true);
	setAcceptTouchEvents(true);
	double offset = snapToPixel(0, crossPen.widthF());
	m_horizontalLine = new QGraphicsLineItem(-100000, offset, 100000, offset, this);
	m_horizontalLine->setPen(crossPen);
	m_verticalLine = new QGraphicsLineItem(offset, -100000, offset, 100000, this);
	m_verticalLine->setPen(crossPen);

	m_selectLineItem = new QGraphicsLineItem(this);
	m_selectLineItem->setVisible(false);
	m_selectLineItem->setPen(selectionPen);
	m_selectRectItem = new QGraphicsRectItem(this);
	m_selectRectItem->setVisible(false);
	m_selectRectItem->setPen(selectionPen);
	m_selectRectItem->setBrush(QColor(255, 255, 0, 64));

	m_textBackgroundItem = new QGraphicsRectItem(this);
	m_textBackgroundItem->setPos(5, 5);
	m_textBackgroundItem->setBrush(textBackgroundBrush);
	m_textBackgroundItem->setPen(Qt::NoPen);
	m_textItem = new QGraphicsSimpleTextItem(this);
	m_textItem->setPos(5, 5);
	m_textItem->setBrush(textBrush);

	setCursor(Qt::BlankCursor);

	connect(transform, SIGNAL(viewChanged()), this, SLOT(updateText()));
}

TrackerItem::~TrackerItem()
{
}

QRectF TrackerItem::boundingRect() const
{
	return QRectF(-100000, -100000, 200000, 200000);
}

void TrackerItem::paint(QPainter *, const QStyleOptionGraphicsItem *, QWidget *)
{
}

Plot2DTransform *TrackerItem::transform() const
{
	return m_transform;
}

void TrackerItem::setTransform(Plot2DTransform *heatmapItem)
{
	assert(heatmapItem);
	disconnect(m_transform, SIGNAL(viewChanged()), this, SLOT(updateText()));
	m_transform = heatmapItem;
	connect(heatmapItem, SIGNAL(viewChanged()), this, SLOT(updateText()));
	updateText();
}

std::function<QString(QPointF)> TrackerItem::hoverLookup() const
{
	return m_hoverLookup;
}

void TrackerItem::setHoverLookup(std::function<QString(QPointF)> hoverLookup)
{
	m_hoverLookup = hoverLookup;
	updateText();
}

TrackerItem::SelectionMode TrackerItem::selectionMode() const
{
	return m_selectionMode;
}

void TrackerItem::setSelectionMode(TrackerItem::SelectionMode selectionMode)
{
	m_selectionMode = selectionMode;
	if (m_selecting) {
		m_selectLineItem->setVisible(selectionMode == LineSelection);
		m_selectRectItem->setVisible(selectionMode == RectSelection);
		m_selecting = (selectionMode != DisableSelection);
	}
}

void TrackerItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
	setPos(event->scenePos());
}

void TrackerItem::hoverEnterEvent(QGraphicsSceneHoverEvent *)
{
	m_horizontalLine->setVisible(true);
	m_verticalLine->setVisible(true);
	m_textBackgroundItem->setVisible(true);
	m_textItem->setVisible(true);
}

void TrackerItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *)
{
	m_horizontalLine->setVisible(false);
	m_verticalLine->setVisible(false);
	m_textBackgroundItem->setVisible(false);
	m_textItem->setVisible(false);
}

void TrackerItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if (m_panning) {
		m_transform->pan(m_panStart, event->scenePos());
	}
	setPos(event->scenePos());
	if (m_selecting) {
		QPointF startLocal = mapFromScene(m_transform->dataToScene(m_selectStart));
		m_selectLineItem->setLine(snapToPixel(QLineF(startLocal, QPointF(0, 0)), m_selectLineItem->pen().widthF()));
		m_selectRectItem->setRect(snapToPixel(QRectF(startLocal, QPointF(0, 0)).normalized(), m_selectRectItem->pen().widthF()));
	}
}

void TrackerItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	QPointF dataPoint = m_transform->sceneToData(event->scenePos());
	if (event->button() == Qt::LeftButton && m_selectionMode != DisableSelection) {
		m_selectStart = dataPoint;
		m_selectLineItem->setLine(QLineF());
		m_selectRectItem->setRect(QRectF());
		m_selectLineItem->setVisible(m_selectionMode == LineSelection);
		m_selectRectItem->setVisible(m_selectionMode == RectSelection);
		m_selecting = true;
	} else if (event->button() == Qt::MiddleButton) {
		m_panStart = dataPoint;
		m_panning = true;
	}
}

void TrackerItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	QPointF dataPoint = m_transform->sceneToData(event->scenePos());
	if (event->button() == Qt::LeftButton) {
		m_selectLineItem->setVisible(false);
		m_selectRectItem->setVisible(false);
		if (m_selecting) {
			m_selecting = false;
			emit selectionComplete(QRectF(m_selectStart, dataPoint));
		}
	} else if (event->button() == Qt::MiddleButton) {
		m_panning = false;
	} else if (event->button() == Qt::RightButton) {
		if (m_selecting) {
			m_selecting = false;
			m_selectLineItem->setVisible(false);
			m_selectRectItem->setVisible(false);
		} else
			emit rightClick(dataPoint);
	}
}

QVariant TrackerItem::itemChange(GraphicsItemChange change, const QVariant &value)
{
	if (change == ItemScenePositionHasChanged)
		updateText();
	return QGraphicsObject::itemChange(change, value);
}

void TrackerItem::updateText()
{
	QPointF dataPos = m_transform->sceneToData(scenePos());
	if (m_hoverLookup)
		m_textItem->setText(m_hoverLookup(dataPos));
	else {
		QPoint roundedPos(int(std::floor(dataPos.x())), int(std::floor(dataPos.y())));
		m_textItem->setText(QString("x: %1 y: %2").arg(roundedPos.x()).arg(roundedPos.y()));
	}
	m_textBackgroundItem->setRect(snapToPixel(m_textItem->boundingRect().adjusted(-2, 0, 2, 0)));
}

} // namespace hpw
