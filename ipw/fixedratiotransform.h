#ifndef HPW_FIXEDRATIOTRANSFORM_H
#define HPW_FIXEDRATIOTRANSFORM_H

#include "ipw/plot2dtransform.h"

namespace ipw
{

/**
 * @brief The FixedRatioTransform class provides a @ref Plot2DTransform with a fixed aspect ratio.
 */
class FixedRatioTransform : public Plot2DTransform
{
public:
	FixedRatioTransform(QObject *parent = 0, double zoom = 1, QPointF origin = QPointF(), QRectF dataRect = QRectF());
	~FixedRatioTransform();

	double zoom() const;
	void setZoom(double zoom, QPointF fixedScenePoint);
	void setView(double zoom, QPointF dataPoint, QPointF scenePoint);

	double minZoom() const;
	double maxZoom() const;
	void setZoomClamp(double minZoom, double maxZoom);

	QRectF dataRect() const;
	void setDataRect(QRectF dataRect);

	QPointF origin() const;
	void setOrigin(QPointF scenePoint);
	void pan(QPointF dataPoint, QPointF scenePoint);

	QSizeF sceneToData(QSizeF sceneSize) const;
	QSizeF dataToScene(QSizeF dataSize) const;

	QPointF sceneToData(QPointF scenePoint) const;
	QPointF dataToScene(QPointF dataPoint) const;

	QRectF sceneToData(QRectF sceneRect) const;
	QRectF dataToScene(QRectF dataRect) const;

private:
	double m_zoom;
	double m_minZoom;
	double m_maxZoom;
	QPointF m_origin;
	QRectF m_dataRect;
	QRectF m_clampRect;
};

} // namespace hpw

#endif // HPW_FIXEDRATIOTRANSFORM_H
