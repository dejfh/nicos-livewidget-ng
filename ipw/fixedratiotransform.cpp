#include "fixedratiotransform.h"

#include "ipw/geometryhelper.h"

namespace ipw
{

FixedRatioTransform::FixedRatioTransform(QObject *parent, double zoom, QPointF origin, QRectF dataRect)
	: Plot2DTransform(parent)
	, m_minZoom(.05)
	, m_maxZoom(50)
	, m_dataRect(dataRect)
{
	setView(zoom, QPointF(), origin);
}

FixedRatioTransform::~FixedRatioTransform()
{
}

double FixedRatioTransform::zoom() const
{
	return m_zoom;
}

void FixedRatioTransform::setZoom(double zoom, QPointF fixedScenePoint)
{
	QPointF dataPoint = sceneToData(fixedScenePoint);
	setView(zoom, dataPoint, fixedScenePoint);
}

void FixedRatioTransform::setView(double zoom, QPointF dataPoint, QPointF scenePoint)
{
	zoom = clamp(zoom, m_minZoom, m_maxZoom);
	m_zoom = zoom;
	QPointF p1 = -toPoint(dataToScene(toSize(m_dataRect.topLeft())));
	QPointF p2 = -toPoint(dataToScene(toSize(m_dataRect.bottomRight())));
	m_clampRect = QRectF(p1, p2).normalized();
	pan(dataPoint, scenePoint);
}

double FixedRatioTransform::minZoom() const
{
	return m_minZoom;
}

double FixedRatioTransform::maxZoom() const
{
	return m_maxZoom;
}

void FixedRatioTransform::setZoomClamp(double minZoom, double maxZoom)
{
	m_minZoom = minZoom;
	m_maxZoom = maxZoom;
}

QRectF FixedRatioTransform::dataRect() const
{
	return m_dataRect;
}

void FixedRatioTransform::setDataRect(QRectF dataRect)
{
	m_dataRect = dataRect;
	setView(m_zoom, QPointF(), m_origin);
}

QPointF FixedRatioTransform::origin() const
{
	return m_origin;
}

void FixedRatioTransform::setOrigin(QPointF scenePoint)
{
	m_origin = clamp(scenePoint, m_clampRect);
	emit viewChanged();
}

void FixedRatioTransform::pan(QPointF dataPoint, QPointF scenePoint)
{
	QSizeF offsetToOrigin = dataToScene(toSize(dataPoint));
	m_origin = scenePoint - toPoint(offsetToOrigin);
	m_origin = clamp(m_origin, m_clampRect);
	emit viewChanged();
}

QSizeF FixedRatioTransform::sceneToData(QSizeF sceneSize) const
{
	return QSizeF(sceneSize.width() / m_zoom, -sceneSize.height() / m_zoom);
}

QSizeF FixedRatioTransform::dataToScene(QSizeF dataSize) const
{
	return QSizeF(dataSize.width() * m_zoom, -dataSize.height() * m_zoom);
}

QPointF FixedRatioTransform::sceneToData(QPointF scenePoint) const
{
	return toPoint(sceneToData(toSize(scenePoint - m_origin)));
}

QPointF FixedRatioTransform::dataToScene(QPointF dataPoint) const
{
	return m_origin + toPoint(dataToScene(toSize(dataPoint)));
}

QRectF FixedRatioTransform::sceneToData(QRectF sceneRect) const
{
	return QRectF(sceneToData(sceneRect.topLeft()), sceneToData(sceneRect.size()));
}

QRectF FixedRatioTransform::dataToScene(QRectF dataRect) const
{
	return QRectF(dataToScene(dataRect.topLeft()), dataToScene(dataRect.size()));
}

} // namespace hpw
