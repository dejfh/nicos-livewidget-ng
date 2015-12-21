#ifndef HPW_GEOMETRYHELPER
#define HPW_GEOMETRYHELPER

#include <algorithm>
#include <QRectF>
#include <QPointF>
#include <QSizeF>
#include <QLineF>

namespace ipw
{

template <typename _T>
inline _T clamp(const _T &value, const _T &min, const _T &max)
{
	return std::min(std::max(value, min), max);
}

inline QPointF clamp(QPointF point, QRectF rect)
{
	rect = rect.normalized();
	return QPointF(clamp(point.x(), rect.left(), rect.right()), clamp(point.y(), rect.top(), rect.bottom()));
}

inline QPointF clamp(QPointF point, QSizeF size)
{
	return clamp(point, QRectF(QPointF(), size));
}

inline QPointF operator+(QPointF point, QSizeF size)
{
	return QPointF(point.x() + size.width(), point.y() + size.height());
}
inline QPointF operator-(QPointF point, QSizeF size)
{
	return QPointF(point.x() - size.width(), point.y() - size.height());
}

inline QSizeF toSize(QPointF point)
{
	return QSizeF(point.x(), point.y());
}
inline QPointF toPoint(QSizeF size)
{
	return QPointF(size.width(), size.height());
}

inline QSizeF operator-(QSizeF size)
{
	return QSizeF(-size.width(), -size.height());
}

bool shouldSnapToHalf(double penWidth);
double snapToPixel(double value, bool snapToHalf = false);
QPointF snapToPixel(QPointF point, bool snapToHalf = false);
QRectF snapToPixel(QRectF rect, bool snapToHalf = false);
QLineF snapToPixel(QLineF line, bool snapToHalf = false);
double snapToPixel(double value, double penWidth);
QPointF snapToPixel(QPointF point, double penWidth);
QRectF snapToPixel(QRectF rect, double penWidth);
QLineF snapToPixel(QLineF line, double penWidth);

} // namespace hpw

#endif // HPW_GEOMETRYHELPER
