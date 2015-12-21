#include "selectionrectitem.h"

#include <algorithm>
#include <cassert>

#include <QPen>
#include <QCursor>

#include "ipw/geometryhelper.h"

#include "safecast.h"

namespace ipw
{

SelectionRectItem::SelectionRectItem(Plot2DTransform *transform, QGraphicsItem *parent)
	: QGraphicsObject(parent)
	, m_transform(transform)
	, m_rectItemBack(new QGraphicsRectItem(this))
	, m_rectItemFront(new QGraphicsRectItem(this))
	, m_dragIndex(NoGrip)
{
	assert(transform);
	setFlag(ItemHasNoContents);

	m_rectItemBack->setPen(QPen(QColor(0, 0, 0, 128), 3, Qt::SolidLine, Qt::FlatCap, Qt::MiterJoin));
	m_rectItemFront->setPen(QPen(QColor(255, 255, 255), 1, Qt::SolidLine, Qt::FlatCap, Qt::MiterJoin));

	for (int grip = 0; grip < GripCount; ++grip) {
		HandleItem *gripItem = new HandleItem(transform, this);
		m_gripItems[grip] = gripItem;
		jfh::assert_result(connect(gripItem, SIGNAL(rightClicked(QPoint)), this, SLOT(gripRightClick(QPoint))));
		jfh::assert_result(connect(gripItem, SIGNAL(dragStarted(QPointF)), this, SLOT(gripDragStart(QPointF))));
		jfh::assert_result(connect(gripItem, SIGNAL(dragContinued(QPointF)), this, SLOT(gripDragContinue(QPointF))));
		jfh::assert_result(connect(gripItem, SIGNAL(dragDone(QPointF)), this, SLOT(gripDragEnd(QPointF))));
		gripItem->setRect(QRectF(-12, -12, 24, 24));
	}
	m_gripItems[TopLeftGrip]->setCursor(Qt::SizeFDiagCursor);
	m_gripItems[TopGrip]->setCursor(Qt::SizeVerCursor);
	m_gripItems[TopRightGrip]->setCursor(Qt::SizeBDiagCursor);
	m_gripItems[LeftGrip]->setCursor(Qt::SizeHorCursor);
	m_gripItems[CenterGrip]->setCursor(Qt::SizeAllCursor);
	m_gripItems[RightGrip]->setCursor(Qt::SizeHorCursor);
	m_gripItems[BottomLeftGrip]->setCursor(Qt::SizeBDiagCursor);
	m_gripItems[BottomGrip]->setCursor(Qt::SizeVerCursor);
	m_gripItems[BottomRightGrip]->setCursor(Qt::SizeFDiagCursor);

	connect(transform, SIGNAL(viewChanged()), this, SLOT(updateGeometry()));
}

SelectionRectItem::~SelectionRectItem()
{
}

QRectF SelectionRectItem::boundingRect() const
{
	return QRectF();
}

void SelectionRectItem::paint(QPainter *, const QStyleOptionGraphicsItem *, QWidget *)
{
}

QRectF SelectionRectItem::rect() const
{
	return m_rect;
}

void SelectionRectItem::setRect(QRectF rect)
{
	m_rect = rect;
	emit selectionChanged(rect);
	updateGeometry();
}

const std::function<void(QPoint)> &SelectionRectItem::rightClickCallback() const
{
	return m_rightClickCallback;
}

void SelectionRectItem::setRightClickCallback(const std::function<void(QPoint)> &callback)
{
	m_rightClickCallback = callback;
}

void SelectionRectItem::updateGeometry()
{
	QRectF sceneRect = m_transform->dataToScene(m_rect);
	sceneRect = snapToPixel(sceneRect, true);
	{
		double horizontal[3] = {sceneRect.left(), (sceneRect.left() + sceneRect.right()) / 2, sceneRect.right()};
		double vertical[3] = {sceneRect.top(), (sceneRect.top() + sceneRect.bottom()) / 2, sceneRect.bottom()};
		for (int y = 0; y < 3; ++y)
			for (int x = 0; x < 3; ++x)
				m_gripItems[x + 3 * y]->setPos(horizontal[x], vertical[y]);
	}
	QCursor cursor1((sceneRect.width() < 0) == (sceneRect.height() < 0) ? Qt::SizeFDiagCursor : Qt::SizeBDiagCursor);
	QCursor cursor2((sceneRect.width() < 0) == (sceneRect.height() < 0) ? Qt::SizeBDiagCursor : Qt::SizeFDiagCursor);

	m_gripItems[TopLeftGrip]->setCursor(cursor1);
	m_gripItems[BottomRightGrip]->setCursor(cursor1);
	m_gripItems[TopRightGrip]->setCursor(cursor2);
	m_gripItems[BottomLeftGrip]->setCursor(cursor2);

	sceneRect = snapToPixel(sceneRect.normalized(), m_rectItemFront->pen().widthF());
	m_rectItemBack->setRect(sceneRect);
	m_rectItemFront->setRect(sceneRect);

	const double gripSize = 24;
	double w = std::max(0.0, sceneRect.width() - gripSize);
	double h = std::max(0.0, sceneRect.height() - gripSize);
	double xc = -w / 2;
	double xl = -gripSize / 2;
	double xr = -gripSize / 2;
	double yc = -h / 2;
	double yt = -gripSize / 2;
	double yb = -gripSize / 2;
	m_gripItems[0]->setRect(QRectF(xl, yt, gripSize, gripSize));
	m_gripItems[1]->setRect(QRectF(xc, yt, w, gripSize));
	m_gripItems[2]->setRect(QRectF(xr, yt, gripSize, gripSize));
	m_gripItems[3]->setRect(QRectF(xl, yc, gripSize, h));
	m_gripItems[4]->setRect(QRectF(xc, yc, w, h));
	m_gripItems[5]->setRect(QRectF(xr, yc, gripSize, h));
	m_gripItems[6]->setRect(QRectF(xl, yb, gripSize, gripSize));
	m_gripItems[7]->setRect(QRectF(xc, yb, w, gripSize));
	m_gripItems[8]->setRect(QRectF(xr, yb, gripSize, gripSize));
}

void SelectionRectItem::gripRightClick(QPoint screenPoint)
{
	if (m_rightClickCallback)
		m_rightClickCallback(screenPoint);
}

void SelectionRectItem::gripDragStart(QPointF)
{
	HandleItem *grip = dynamic_cast<HandleItem *>(QObject::sender());
	int i;
	for (i = 0; i < 8; ++i)
		if (m_gripItems[i] == grip)
			break;
	m_dragIndex = i;
}

void SelectionRectItem::gripDragContinue(QPointF dataPoint)
{
	if (m_dragIndex == CenterGrip)
		m_rect.moveCenter(dataPoint);
	if (m_dragIndex / 3 == 0)
		m_rect.setTop(dataPoint.y());
	else if (m_dragIndex / 3 == 2)
		m_rect.setBottom(dataPoint.y());
	if (m_dragIndex % 3 == 0)
		m_rect.setLeft(dataPoint.x());
	else if (m_dragIndex % 3 == 2)
		m_rect.setRight(dataPoint.x());
	emit selectionChanged(m_rect);
	updateGeometry();
}

void SelectionRectItem::gripDragEnd(QPointF)
{
	m_dragIndex = NoGrip;
}

} // namespace hpw
