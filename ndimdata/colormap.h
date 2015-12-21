#ifndef NDIMDATA_COLORMAP_H
#define NDIMDATA_COLORMAP_H

#include <QRgb>
#include <cmath>
#include <qwt_color_map.h>

namespace ndimdata {

struct IntervalScaler {
	double summand;
	double factor;
	double min;
	double max;

	inline IntervalScaler() {}
	inline IntervalScaler(double summand, double factor, double min, double max)
		: summand(summand)
		, factor(factor)
		, min(std::min(min, max))
		, max(std::max(min, max))
	{
	}

	static inline IntervalScaler fromBounds(double min_in, double max_in, double min_out, double max_out)
	{
		IntervalScaler scaler;
		scaler.factor = (max_out - min_out) / (max_in - min_in);
		scaler.min = std::min(min_out, max_out);
		scaler.max = std::max(min_out, max_out);
		scaler.summand = -min_in + (min_out / scaler.factor);
		return scaler;
	}

	static inline IntervalScaler fromBoundsAndRange(double min_in, double max_in, double range)
	{
		return fromBounds(min_in, max_in, 0, range); //
	}

	inline double scale(double value) const { return std::min(std::max((value + summand) * factor, min), max); }
};

struct ColorMapGrayscale {
	IntervalScaler scaler;

	ColorMapGrayscale(double from, double to)
		: scaler(-from, 256.0 / (to - from), 0, 255)
	{
	}

	inline QRgb operator()(double value) const
	{
		int i = int(scaler.scale(value));
		return qRgb(i, i, i);
	}
};

struct ColorMapColor {
	static inline QRgb fromAngle(double angle)
	{
		static const double sin120 = 0.86602540378443864676372317075294;
		static const double cos120 = -0.5;

		static const double r0 = .5 * 255;
		static const double drdx = r0 * 1; // cos(0°)
		static const double drdy = r0 * 0; // sin(0°)

		static const double g0 = .5 * 255;
		static const double dgdx = g0 * cos120; // cos(120°)
		static const double dgdy = g0 * sin120; // sin(120°)

		static const double b0 = .5 * 255;
		static const double dbdx = b0 * cos120;  // cos(240°)
		static const double dbdy = b0 * -sin120; // sin(240°)

		double x = std::cos(angle);
		double y = std::sin(angle);

		double r = .5 + r0 + x * drdx + y * drdy;
		double g = .5 + g0 + x * dgdx + y * dgdy;
		double b = .5 + b0 + x * dbdx + y * dbdy;

		return qRgb(int(r), int(g), int(b));
	}

	IntervalScaler scaler;

	ColorMapColor(double from, double to, double from_color = 240, double to_color = 0)
	{
		static const double pi = 3.1415926535897932384626433832795;

		scaler = IntervalScaler::fromBounds(from, to, from_color * (pi / 180), to_color * (pi / 180));
	}

	inline QRgb operator()(double value) const { return fromAngle(scaler.scale(value)); }
};

struct ColorMapCyclic : ColorMapColor {
	ColorMapCyclic(double from, double to)
		: ColorMapColor(from, to, 240, -120)
	{
	}
};

template <typename _T_colorMap>
class ColorMapWrapper : public QwtColorMap
{
	bool invert;

public:
	ColorMapWrapper(bool invert = false)
		: invert(invert)
	{
	}
	virtual ~ColorMapWrapper() {}

	virtual QRgb rgb(const QwtInterval &interval, double value) const
	{
		if (!invert) {
			_T_colorMap colorMap(interval.minValue(), interval.maxValue());
			return colorMap(value);
		} else {
			_T_colorMap colorMap(interval.maxValue(), interval.minValue());
			return colorMap(value);
		}
	}

	virtual unsigned char colorIndex(const QwtInterval &, double) const { return 0; }
};

} // namespace ndimdata

#endif // NDIMDATA_COLORMAP_H
