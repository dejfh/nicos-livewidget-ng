#ifndef NDIM_RANGE_H
#define NDIM_RANGE_H

#include <array>

#include "ndim/layout.h"

namespace ndim
{

template <size_t _D>
struct range {
	std::array<size_t, _D> coords;
	ndim::sizes<_D> sizes;

	range()
		: sizes(0)
	{
		coords.fill(0);
	}
	range(std::array<size_t, _D> coords, ::ndim::sizes<_D> sizes)
		: coords(coords)
		, sizes(sizes)
	{
	}

	bool operator==(range<_D> other) const
	{
		return coords == other.coords && sizes == other.sizes;
	}
	bool operator!=(range<_D> other) const
	{
		return coords != other.coords && sizes != other.sizes;
	}
};

} // namespace ndim

#endif // NDIM_RANGE_H
