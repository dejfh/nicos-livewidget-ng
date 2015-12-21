#include "griditem.h"

#include <algorithm>
#include <cmath>
#include <cassert>

#include <QPainter>
#include <QGraphicsView>
#include <QStyleOptionGraphicsItem>

#include "geometryhelper.h"

namespace ipw
{

GridItem::GridItem(Plot2DTransform *transform, int minDistance, const QPen &pen, QGraphicsItem *parent)
	: QGraphicsObject(parent)
	, m_transform(transform)
	, m_minDistance(minDistance)
	, m_pen(pen)
	, m_manualInterval(1, 1)
	, m_useManualInterval(false)
{
	assert(transform);
	this->setFlag(ItemUsesExtendedStyleOption);
	connect(transform, SIGNAL(viewChanged()), this, SLOT(updateToScale()));
}

GridItem::~GridItem()
{
}

Plot2DTransform *GridItem::transform() const
{
	return m_transform;
}

int GridItem::minDistance() const
{
	return m_minDistance;
}

void GridItem::setMinDistance(int distance)
{
	m_minDistance = distance;
	this->update();
}

bool GridItem::useManualInterval() const
{
	return m_useManualInterval;
}

void GridItem::setUseManualInterval(bool use)
{
	m_useManualInterval = use;
	this->update();
}

QSizeF GridItem::manualInterval() const
{
	return m_manualInterval;
}

void GridItem::setManualInterval(double interval)
{
	m_manualInterval = QSizeF(interval, interval);
	this->update();
}
void GridItem::setManualInterval(QSizeF interval)
{
	m_manualInterval = interval;
	this->update();
}

const QPen &GridItem::pen() const
{
	return m_pen;
}

void GridItem::setPen(const QPen &pen)
{
	m_pen = pen;
	this->update();
}

QRectF GridItem::boundingRect() const
{
	return QRectF(-10000, -10000, 20000, 20000);
}

void GridItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
{
	assert(option);
	QRectF sceneRect = option->exposedRect.normalized();
	double penAdjust = m_pen.widthF() / 2 + 1;
	sceneRect.adjust(-penAdjust, -penAdjust, penAdjust, penAdjust);
	painter->setPen(m_pen);

	bool snapToHalf = shouldSnapToHalf(m_pen.widthF());

	QRectF intervalRect = firstSceneInterval(sceneRect);
	for (double y = intervalRect.top(); y <= sceneRect.bottom(); y += intervalRect.height()) {
		double sy = snapToPixel(y, snapToHalf);
		painter->drawLine(QLineF(-10000, sy, 10000, sy));
	}
	for (double x = intervalRect.left(); x <= sceneRect.right(); x += intervalRect.width()) {
		double sx = snapToPixel(x, snapToHalf);
		painter->drawLine(QLineF(sx, -10000, sx, 10000));
	}
}

/**
 * @brief niceInterval Rounds the value to the next power of ten, two times a power of ten, or five times a power of ten.
 */
double niceInterval(double interval)
{
	double logInterval = std::log10(std::fabs(interval));
	int magnitude = int(std::floor(logInterval));
	double remainder = logInterval - magnitude;
	interval = std::pow(10, magnitude);
	if (remainder >= std::log10(5))
		interval *= 5;
	else if (remainder >= std::log10(2))
		interval *= 2;
	return interval;
}

QSizeF niceInterval(QSizeF interval)
{
	return QSizeF(niceInterval(interval.width()), niceInterval(interval.height()));
}

double intervalStart(double lowerBound, double interval, double origin)
{
	double offset = fmod(origin - lowerBound, interval);
	if (offset < 0)
		offset += interval;
	return lowerBound + offset;
}

QPointF intervalStart(QPointF lowerBound, QSizeF interval, QPointF origin)
{
	return QPointF(intervalStart(lowerBound.x(), interval.width(), origin.x()), intervalStart(lowerBound.y(), interval.height(), origin.y()));
}

/**
 * @brief GridItem::firstDataInterval Determins the first grid interval in data space.
 */
QRectF GridItem::firstDataInterval(QRectF dataRect) const
{
	QSizeF interval(m_minDistance, m_minDistance);
	interval = m_transform->sceneToData(interval);
	interval = niceInterval(interval);
	if (m_useManualInterval) {
		interval.setWidth(std::max(m_manualInterval.width(), interval.width()));
		interval.setHeight(std::max(m_manualInterval.height(), interval.height()));
	}
	QPointF start = intervalStart(dataRect.normalized().topLeft(), interval, QPointF());
	return QRectF(start, interval);
}

/**
 * @brief GridItem::firstSceneInterval Determins the first grid interval in scene space
 */
QRectF GridItem::firstSceneInterval(QRectF sceneRect) const
{
	QSizeF interval(m_minDistance, m_minDistance);
	interval = m_transform->sceneToData(interval);
	interval = niceInterval(interval);
	if (m_useManualInterval) {
		interval.setWidth(std::max(m_manualInterval.width(), interval.width()));
		interval.setHeight(std::max(m_manualInterval.height(), interval.height()));
	}
	interval = m_transform->dataToScene(interval);
	interval = QSizeF(std::fabs(interval.width()), std::fabs(interval.height()));
	QPointF start = intervalStart(sceneRect.normalized().topLeft(), interval, m_transform->origin());
	return QRectF(start, interval);
}

void GridItem::updateToScale()
{
	this->update();
}

} // namespace hpw
