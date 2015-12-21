#include "ipw/geometryhelper.h"

#include <cmath>

/*
 * Helper functions to position lines and rectangles on device pixels.
 * */

namespace ipw
{

bool shouldSnapToHalf(double penWidth)
{
	if (penWidth <= 0)
		return false;
	else if (penWidth < 1)
		return true;
	else if (std::fmod(penWidth, 2) >= 1)
		return true;
	return false;
}

double snapToPixel(double value, bool snapToHalf)
{
	if (snapToHalf)
		return std::floor(value) + .5;
	else
		return std::floor(value + .5);
}

QPointF snapToPixel(QPointF point, bool snapToHalf)
{
	return QPointF(snapToPixel(point.x(), snapToHalf), snapToPixel(point.y(), snapToHalf));
}

QRectF snapToPixel(QRectF rect, bool snapToHalf)
{
	return QRectF(snapToPixel(rect.topLeft(), snapToHalf), snapToPixel(rect.bottomRight(), snapToHalf));
}

QLineF snapToPixel(QLineF line, bool snapToHalf)
{
	return QLineF(snapToPixel(line.p1(), snapToHalf), snapToPixel(line.p2(), snapToHalf));
}

double snapToPixel(double value, double penWidth)
{
	return snapToPixel(value, shouldSnapToHalf(penWidth));
}

QPointF snapToPixel(QPointF point, double penWidth)
{
	return snapToPixel(point, shouldSnapToHalf(penWidth));
}

QLineF snapToPixel(QLineF line, double penWidth)
{
	return snapToPixel(line, shouldSnapToHalf(penWidth));
}

QRectF snapToPixel(QRectF rect, double penWidth)
{
	return snapToPixel(rect, shouldSnapToHalf(penWidth));
}

} // namespace hpw
