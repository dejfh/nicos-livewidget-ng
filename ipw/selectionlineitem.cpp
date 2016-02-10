#include "selectionlineitem.h"

#include <algorithm>
#include <cassert>

#include <QPen>
#include <QCursor>

#include "ipw/geometryhelper.h"

#include "safecast.h"

namespace ipw
{

SelectionLineItem::SelectionLineItem(Plot2DTransform *transform, QGraphicsItem *parent)
	: QGraphicsObject(parent)
	, m_transform(transform)
	, m_lineItemBack(new QGraphicsLineItem(this))
	, m_lineItemFront(new QGraphicsLineItem(this))
	, m_dragIndex(NoGrip)
{
	assert(transform);
	setFlag(ItemHasNoContents);

	m_lineItemBack->setPen(QPen(QColor(0, 0, 0, 128), 3, Qt::SolidLine, Qt::RoundCap, Qt::MiterJoin));
	m_lineItemFront->setPen(QPen(QColor(255, 255, 255), 1, Qt::SolidLine, Qt::RoundCap, Qt::MiterJoin));

	for (int grip = 0; grip < GripCount; ++grip) {
		HandleItem *gripItem = new HandleItem(transform, this);
		m_gripItems[grip] = gripItem;
		jfh::assert_result(connect(gripItem, SIGNAL(rightClicked(QPoint)), this, SLOT(gripRightClick(QPoint))));
		jfh::assert_result(connect(gripItem, SIGNAL(dragStarted(QPointF)), this, SLOT(gripDragStart(QPointF))));
		jfh::assert_result(connect(gripItem, SIGNAL(dragContinued(QPointF)), this, SLOT(gripDragContinue(QPointF))));
		jfh::assert_result(connect(gripItem, SIGNAL(dragDone(QPointF)), this, SLOT(gripDragEnd(QPointF))));
		gripItem->setRect(QRectF(-12, -12, 24, 24));
	}
	m_gripItems[StartGrip]->setCursor(Qt::SizeAllCursor);
	m_gripItems[EndGrip]->setCursor(Qt::SizeAllCursor);
	m_gripItems[MidGrip]->setCursor(Qt::SizeAllCursor);

	connect(transform, SIGNAL(viewChanged()), this, SLOT(updateGeometry()));
}

SelectionLineItem::~SelectionLineItem()
{
}

QRectF SelectionLineItem::boundingRect() const
{
	return QRectF();
}

void SelectionLineItem::paint(QPainter *, const QStyleOptionGraphicsItem *, QWidget *)
{
}

QLineF SelectionLineItem::line() const
{
	return m_line;
}

void SelectionLineItem::setLine(QLineF line)
{
	m_line = line;
	emit selectionChanged(m_line);
	updateGeometry();
}

void SelectionLineItem::updateGeometry()
{
	QLineF sceneLine(m_transform->dataToScene(m_line.p1()), m_transform->dataToScene(m_line.p2()));
	sceneLine = snapToPixel(sceneLine, true);
	m_gripItems[StartGrip]->setPos(sceneLine.p1());
	m_gripItems[EndGrip]->setPos(sceneLine.p2());
	m_gripItems[MidGrip]->setPos((sceneLine.p1() + sceneLine.p2()) / 2);

	m_lineItemBack->setLine(sceneLine);
	m_lineItemFront->setLine(sceneLine);

	const double gripSize = 24;
	double angle = -sceneLine.angle();
	double w = std::max(0.0, sceneLine.length() - gripSize);
	m_gripItems[MidGrip]->setRect(QRectF(-w / 2, -gripSize / 2, w, gripSize));
	m_gripItems[MidGrip]->setRotation(angle);
	//	m_gripItems[StartGrip]->setRect(QRectF(-w / 2, -gripSize / 2, w, gripSize / 2));
	m_gripItems[StartGrip]->setRotation(angle);
	//	m_gripItems[EndGrip]->setRect(QRectF(-w / 2, -gripSize / 2, w, gripSize / 2));
	m_gripItems[EndGrip]->setRotation(angle);
}

void SelectionLineItem::gripRightClick(QPoint screenPoint)
{
	emit rightClicked(m_line, screenPoint);
}

void SelectionLineItem::gripDragStart(QPointF)
{
	HandleItem *grip = dynamic_cast<HandleItem *>(QObject::sender());
	int i;
	for (i = 0; i < 8; ++i)
		if (m_gripItems[i] == grip)
			break;
	m_dragIndex = i;
}

void SelectionLineItem::gripDragContinue(QPointF dataPoint)
{
	switch (m_dragIndex) {
	case StartGrip:
		m_line.setP1(dataPoint);
		break;
	case EndGrip:
		m_line.setP2(dataPoint);
		break;
	case MidGrip: {
		QPointF diff = (m_line.p2() - m_line.p1()) / 2;
		m_line.setP1(dataPoint - diff);
		m_line.setP2(dataPoint + diff);
		break;
	}
	default:
		break;
	}
	emit selectionChanged(m_line);
	updateGeometry();
}

void SelectionLineItem::gripDragEnd(QPointF)
{
	m_dragIndex = NoGrip;
}

} // namespace hpw
