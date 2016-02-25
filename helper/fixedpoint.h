#ifndef HELPER_FIXEDPOINT_H
#define HELPER_FIXEDPOINT_H

#include <cstddef>

namespace hlp
{

template <size_t _denominator>
struct FixedPoint {
	long numerator;
	static const size_t denominator = _denominator;

	FixedPoint()
	{
	}
	FixedPoint(float value)
		: numerator(long(value *_denominator))
	{
	}
	explicit FixedPoint(double value)
		: numerator(long(value *_denominator))
	{
	}

	operator float() const
	{
		return float(numerator) / _denominator;
	}
	operator double() const
	{
		return float(numerator) / _denominator;
	}
};

template <size_t _denominator>
FixedPoint<_denominator> &operator+=(FixedPoint<_denominator> &a, FixedPoint<_denominator> b)
{
	a.numerator += b.numerator;
	return a;
}
template <size_t _denominator>
FixedPoint<_denominator> &operator-=(FixedPoint<_denominator> &a, FixedPoint<_denominator> b)
{
	a.numerator -= b.numerator;
	return a;
}

template <size_t _denominator>
FixedPoint<_denominator> operator+(FixedPoint<_denominator> a, FixedPoint<_denominator> b)
{
	a.numerator += b.numerator;
	return a;
}
template <size_t _denominator>
FixedPoint<_denominator> operator-(FixedPoint<_denominator> a, FixedPoint<_denominator> b)
{
	a.numerator -= b.numerator;
	return a;
}

} // namespace hlp

#endif // HELPER_FIXEDPOINT_H
