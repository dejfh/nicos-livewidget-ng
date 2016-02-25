#ifndef TOMO_AXISOFROTATION_H
#define TOMO_AXISOFROTATION_H

#include <stdint.h>

#include "ndim/pointer.h"

#include <QPointF>

namespace tomo
{

struct AxisOfRotation {
	size_t width;
	size_t height;
	double center_layer_0;
	double tan;			  // dy / dx (on layer) == -dx / dy (on axis)
	double cos_square;	// dy / dlayer (on axis)
	double sin_times_cos; // -dx / dlayer (on axis)

	inline double correct(double x) const
	{
		return (x - width * .5) * tan;
	}
	inline double correct(double x, double y) const
	{
		return y + correct(x);
	}
	inline QPointF correct(QPointF &point) const
	{
		return QPointF(point.x(), correct(point.x(), point.y()));
	}

	inline double invCorrect(double x) const
	{
		return -correct(x);
	}
	inline double invCorrect(double x, double y) const
	{
		return y + invCorrect(x);
	}
	inline QPointF invCorrect(QPointF &point) const
	{
		return QPointF(point.x(), invCorrect(point.x(), point.y()));
	}

	inline std::intptr_t layerOf(double x, double y) const
	{
		return std::intptr_t(correct(x, y));
	}
	inline std::intptr_t layerOf(QPointF &point) const
	{
		return layerOf(point.x(), point.y());
	}

	inline double yOfLayer(double x, size_t layer) const
	{
		return correct(x, layer + .5);
	}
	inline QPointF pointOfLayer(double x, size_t layer) const
	{
		return QPointF(x, yOfLayer(x, layer));
	}

	inline double axisXOfLayer(size_t layer) const
	{
		return center_layer_0 - layer * sin_times_cos;
	}
	inline QPointF axisPointOfLayer(size_t layer) const
	{
		return pointOfLayer(layer, axisXOfLayer(layer));
	}

	inline double axisXOfEdge() const
	{
		return center_layer_0 + yOfLayer(center_layer_0, 0) * tan;
	}

	inline AxisOfRotation()
	{
	}

private:
	inline AxisOfRotation(size_t width, size_t height, double center_layer_0, double tan)
		: width(width)
		, height(height)
		, center_layer_0(center_layer_0)
		, tan(tan)
		, cos_square(1.0 / (tan * tan + 1))
		, sin_times_cos(cos_square * tan)
	{
	}

public:
	static inline AxisOfRotation fromLayer(size_t width, size_t height, double center_layer_0, double tan)
	{
		return AxisOfRotation(width, height, center_layer_0, tan);
	}

	static inline AxisOfRotation fromEdge(size_t width, size_t height, double center_edge, double tan)
	{
		AxisOfRotation axis(width, height, center_edge, tan);
		double y_edge = axis.yOfLayer(center_edge, 0);
		axis.center_layer_0 = center_edge - y_edge * axis.sin_times_cos;
		return axis;
	}
};

} // namespace tomo

#endif // TOMO_AXISOFROTATION_H
