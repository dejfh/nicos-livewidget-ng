#ifndef NDIM_LAYOUT_H
#define NDIM_LAYOUT_H

#include <cassert>
#include <array>
#include <cstddef>
#include <numeric>

#include <initializer_list>

#include "helper/array.h"

namespace ndim
{

template <size_t Dimensionality>
using Indices = hlp::array::Indices<Dimensionality>;

template <size_t Dimensionality>
using Sizes = std::array<size_t, Dimensionality>;

template <size_t Dimensionality>
using Strides = std::array<ptrdiff_t, Dimensionality>;

template <size_t Dimensionality>
size_t totalCount(Sizes<Dimensionality> sizes)
{
	size_t count = 1;
	for (size_t size : sizes)
		count *= size;
	return count;
}

template <size_t Dimensionality>
bool isEmpty(Sizes<Dimensionality> sizes)
{
	for (size_t size : sizes)
		if (size == 0)
			return true;
	return false;
}

template <size_t Dimensionality>
bool contains(Sizes<Dimensionality> sizes, Indices<Dimensionality> coords)
{
	for (auto itSizes = sizes.cbegin(), endSizes = sizes.cend(), itCoords = coords.cbegin(); itSizes != endSizes; ++itSizes, ++itCoords)
		if (*itSizes <= *itCoords)
			return false;
	return true;
}
template <size_t Dimensionality, typename... IndexTypes>
bool contains(Sizes<Dimensionality> sizes, size_t coord0, IndexTypes... coordN)
{
	return contains(sizes, Indices<Dimensionality>{coord0, coordN...});
}

template <size_t Dimensionality>
bool isVirtual(Strides<Dimensionality> strides)
{
	for (ptrdiff_t stride : strides)
		if (stride == 0)
			return true;
	return false;
}

template <size_t _D>
struct sizes : std::array<size_t, _D> {
	sizes()
	{
	}
	sizes(size_t value)
	{
		std::array<size_t, _D>::fill(value);
	}
	sizes(const std::array<size_t, _D> &other)
		: std::array<size_t, _D>(other)
	{
	}
	bool operator==(const std::array<size_t, _D> &other)
	{
		return static_cast<std::array<size_t, _D> &>(*this) == other;
	}
	bool operator!=(const std::array<size_t, _D> &other)
	{
		return static_cast<std::array<size_t, _D> &>(*this) != other;
	}

	size_t size() const
	{
		size_t s = 1;
		for (size_t d = 0; d < _D; ++d)
			s *= (*this)[d];
		return s;
	}
	bool isEmpty() const
	{
		for (size_t d = 0; d < _D; ++d)
			if ((*this)[d] == 0)
				return true;
		return false;
	}

	bool contains(Indices<_D> coords)
	{
		for (size_t i = 0; i < _D; ++i)
			if (coords[i] >= (*this)[i])
				return false;
		return true;
	}

	template <typename... coords_t>
	bool contains(size_t coordinate, coords_t... coordinates)
	{
		std::array<size_t, _D> coords = {coordinate, size_t(coordinates)...};
		return contains(coords);
	}

	sizes<_D - 1> removeDimension(size_t dimension) const
	{
		return hlp::array::remove(*this, dimension);
	}

	sizes<_D + 1> addDimension(size_t dimension, size_t insertedSize) const
	{
		return hlp::array::insert(*this, dimension, insertedSize);
	}

	template <size_t _SD>
	sizes<_SD> selectDimensions(Indices<_SD> dimensions) const
	{
		return hlp::array::select(*this, dimensions);
	}

	template <size_t _AD>
	sizes<_D + _AD> addDimensions(sizes<_AD> insertedSizes, Indices<_AD> insertedDimensions) const
	{
		return hlp::array::insert(*this, insertedSizes, insertedDimensions);
	}
};

template <size_t _D>
struct strides;

template <size_t _D>
size_t _indexOf_forward(const strides<_D> &);
template <size_t _D, typename... indices_t>
size_t _indexOf_forward(const strides<_D> &strides, size_t coordinate, indices_t... moreCoordinates);

template <size_t _D>
struct strides : std::array<ptrdiff_t, _D> {
	strides()
	{
	}
	strides(const std::array<ptrdiff_t, _D> &other)
		: std::array<ptrdiff_t, _D>(other)
	{
	}
	bool operator==(const std::array<ptrdiff_t, _D> &other)
	{
		return static_cast<std::array<ptrdiff_t, _D> &>(*this) == other;
	}
	bool operator!=(const std::array<ptrdiff_t, _D> &other)
	{
		return static_cast<std::array<ptrdiff_t, _D> &>(*this) != other;
	}

	template <typename... indices_t>
	size_t indexOf(size_t coordinate, indices_t... moreCoordinates) const
	{
		static_assert(sizeof...(indices_t) + 1 == _D, "count of indices does not match count of dimensions.");
		return _indexOf_forward(*this, coordinate, moreCoordinates...);
	}

	size_t indexOf(std::array<size_t, _D> coordinates) const
	{
		size_t pos = 0;
		for (size_t d = 0; d < _D; ++d)
			pos += (*this)[d] * coordinates[d];
		return pos;
	}

	strides<_D - 1> removeDimension(size_t dimension) const
	{
		return hlp::array::remove(*this, dimension);
	}

	strides<_D + 1> addDimension(size_t dimension, size_t insertedStride = 0) const
	{
		return hlp::array::insert(*this, dimension, insertedStride);
	}

	template <size_t _SD>
	strides<_SD> selectDimensions(std::array<size_t, _SD> dimensions) const
	{
		return hlp::array::select(*this, dimensions);
	}
};

template <size_t _D>
size_t _indexOf_forward(const ndim::strides<_D> &)
{
	return 0;
}
template <size_t _D, typename... indices_t>
size_t _indexOf_forward(const ndim::strides<_D> &strides, size_t coordinate, indices_t... moreCoordinates)
{
	return strides[_D - 1 - sizeof...(indices_t)] * coordinate + _indexOf_forward(strides, moreCoordinates...);
}

template <size_t _D>
struct layout {
	ndim::sizes<_D> sizes;
	ndim::strides<_D> strides;

	layout()
	{
	}
	explicit layout(ndim::sizes<_D> sizes)
		: sizes(sizes)
	{
		ptrdiff_t stride = 1;
		for (size_t d = 0; d < _D; ++d) {
			strides[d] = stride;
			stride *= sizes[d];
		}
	}
	layout(ndim::sizes<_D> sizes, ndim::strides<_D> strides)
		: sizes(sizes)
		, strides(strides)
	{
	}

	template <typename... coordinatess_t>
	size_t indexOf(size_t coordinate, coordinatess_t... moreCoordinates) const
	{
		static_assert(sizeof...(coordinatess_t) + 1 == _D, "count of coordinates does not match count of dimensions.");
		return strides.indexOf(coordinate, moreCoordinates...);
	}

	template <typename... coordinatess_t>
	size_t operator()(size_t coordinate, coordinatess_t... moreCoordinates) const
	{
		static_assert(sizeof...(coordinatess_t) + 1 == _D, "count of coordinates does not match count of dimensions.");
		return strides.indexOf(coordinate, moreCoordinates...);
	}

	size_t indexOf(const std::array<size_t, _D> &coordinates) const
	{
		return strides.indexOf(coordinates);
	}

	size_t operator[](const std::array<size_t, _D> &coordinates) const
	{
		return strides.indexOf(coordinates);
	}

	size_t width() const
	{
		static_assert(_D >= 1, "dimension to small for width");
		return sizes[0];
	}
	size_t height() const
	{
		static_assert(_D >= 2, "dimension to small for height");
		return sizes[1];
	}
	size_t depth() const
	{
		static_assert(_D >= 3, "dimension to small for depth");
		return sizes[2];
	}

	size_t &width()
	{
		static_assert(_D >= 1, "dimension to small for width");
		return sizes[0];
	}
	size_t &height()
	{
		static_assert(_D >= 2, "dimension to small for height");
		return sizes[1];
	}
	size_t &depth()
	{
		static_assert(_D >= 3, "dimension to small for depth");
		return sizes[2];
	}

	size_t size() const
	{
		return sizes.size();
	}
	bool isEmpty() const
	{
		return sizes.isEmpty();
	}
	bool isContiguous() const
	{

		ptrdiff_t stride = 1;
		for (size_t d = 0; d < _D; ++d) {
			if (stride != strides[d])
				return false;
			stride *= sizes[d];
		}
		return true;
	}

	layout<_D - 1> removeDimension(size_t dimension) const
	{
		return layout<_D - 1>(hlp::array::remove(sizes, dimension), hlp::array::remove(strides, dimension));
	}

	layout<_D + 1> addDimension(size_t dimension, size_t virtualSize) const
	{
		return layout<_D + 1>(hlp::array::insert(sizes, dimension, virtualSize), hlp::array::insert(strides, dimension, 0));
	}

	template <size_t _AD>
	layout<_D + _AD> addDimensions(Indices<_AD> dimensions, Sizes<_AD> virtualSizes) const
	{
		return layout<_D + _AD>(hlp::array::insert(sizes, virtualSizes, dimensions), hlp::array::insert(strides, dimensions, ptrdiff_t(0)));
	}

	template <size_t _SD>
	layout<_SD> selectDimensions(std::array<size_t, _SD> dimensions) const
	{
		return layout<_SD>(sizes.selectDimensions(dimensions), strides.selectDimensions(dimensions));
	}

	std::array<ptrdiff_t, _D> getHops() const
	{
		std::array<ptrdiff_t, _D> hops;
		hops[0] = strides[0];
		for (size_t d = 1; d < _D; ++d)
			hops[d] = strides[d] - strides[d - 1] * sizes[d - 1];
		return hops;
	}
};

template <typename... SizesTypes>
layout<1 + sizeof...(SizesTypes)> make_layout_contiguous(size_t size0, SizesTypes... sizeN)
{
	return layout<1 + sizeof...(SizesTypes)>(ndim::Sizes<1 + sizeof...(SizesTypes)>{size0, sizeN...});
}

// template <size_t _D>
// void _make_layout_forward(layout<_D> &)
//{
//}
// template <size_t _D, typename... sizes_and_strides_t>
// void _make_layout_forward(layout<_D> &layout, size_t size, size_t stride, sizes_and_strides_t... more_sizes_and_strides)
//{
//	layout.sizes[_D - sizeof...(sizes_and_strides_t) / 2 - 1] = size;
//	layout.strides[_D - sizeof...(sizes_and_strides_t) / 2 - 1] = stride;
//	_make_layout_forward(layout, more_sizes_and_strides...);
//}
// template <typename... sizes_and_strides_t>
// layout<sizeof...(sizes_and_strides_t) / 2 + 1> make_layout(size_t size, size_t stride, sizes_and_strides_t... more_sizes_and_strides)
//{
//	layout<sizeof...(sizes_and_strides_t) / 2 + 1> layout;
//	_make_layout_forward(layout, size, stride, more_sizes_and_strides...);
//	return layout;
//}

} // namespace ndim

#endif // NDIM_LAYOUT_H
