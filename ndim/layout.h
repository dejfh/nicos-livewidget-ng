#ifndef NDIM_LAYOUT_H
#define NDIM_LAYOUT_H

#include <cassert>
#include <array>
#include <cstddef>
#include <numeric>
#include <exception>

#include <initializer_list>

#include "helper/helper.h"
#include "helper/array.h"
#include "helper/byteoffset.h"

#include <iostream>

namespace ndim
{

template <size_t Dimensionality>
using Indices = hlp::array::Indices<Dimensionality>;

template <size_t Dimensionality>
using Sizes = std::array<size_t, Dimensionality>;

template <size_t Dimensionality>
using Strides = std::array<hlp::byte_offset_t, Dimensionality>;

using IndicesVar = std::vector<size_t>;
using ShapeVar = std::vector<size_t>;
using StridesVar = std::vector<hlp::byte_offset_t>;

inline Indices<0> makeIndices()
{
	return Indices<0>();
}
template <typename... IndexTypes>
Indices<sizeof...(IndexTypes)> makeIndices(IndexTypes... indices)
{
	return Indices<sizeof...(IndexTypes)>{size_t(indices)...};
}
inline Sizes<0> makeSizes()
{
	return Sizes<0>();
}
template <typename... SizeTypes>
Sizes<sizeof...(SizeTypes)> makeSizes(SizeTypes... sizes)
{
	return Sizes<sizeof...(SizeTypes)>{size_t(sizes)...};
}
inline Strides<0> makeStrides()
{
	return Strides<0>();
}
template <typename... StrideTypes>
Strides<sizeof...(StrideTypes)> makeStrides(StrideTypes... strides)
{
    return Strides<sizeof...(StrideTypes)>{hlp::byte_offset_t(strides)...};
}

template <size_t Dimensionality>
size_t totalCount(Sizes<Dimensionality> sizes)
{
	size_t count = 1;
	for (size_t size : sizes)
		count *= size;
	return count;
}
inline size_t totalCount(const ShapeVar &sizes)
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
inline bool isEmpty(const ShapeVar &sizes)
{
    for (size_t size : sizes)
        if (size == 0)
            return true;
    return false;
}

template <size_t Dimensionality>
bool contains(Sizes<Dimensionality> shape, Indices<Dimensionality> coords)
{
    for (auto itShape = shape.cbegin(), endShape = shape.cend(), itCoords = coords.cbegin(); itShape != endShape; ++itShape, ++itCoords)
        if (*itShape <= *itCoords)
			return false;
	return true;
}
inline bool contains(const ShapeVar &shape, const IndicesVar coords)
{
    for (auto itShape = shape.cbegin(), endShape = shape.cend(), itCoords = coords.cbegin(), endCoords = coords.cend();
         itShape != endShape && itCoords != endCoords; ++itShape, ++itCoords)
        if (*itShape <= *itCoords)
            return false;
    return true;
}

template <size_t Dimensionality, typename... IndexTypes>
bool contains(Sizes<Dimensionality> shape, size_t coord0, IndexTypes... coordN)
{
    return contains(shape, Indices<Dimensionality>{coord0, coordN...});
}

template <size_t Dimensionality>
bool isVirtual(Strides<Dimensionality> strides)
{
    for (hlp::byte_offset_t stride : strides)
        if (!stride)
			return true;
	return false;
}
inline bool isVirtual(const StridesVar &strides)
{
    for (auto stride : strides)
        if (!stride)
            return true;
    return false;
}

template <size_t Dimensionality>
hlp::byte_offset_t offsetOf(Strides<Dimensionality> strides, Indices<Dimensionality> coords)
{
    hlp::byte_offset_t offset(0);
    for (auto itStrides = strides.cbegin(), endStrides = strides.cend(), itCoords = coords.cbegin(); itStrides != endStrides; ++itStrides, ++itCoords)
        offset += *itStrides * *itCoords;
    return offset;
}
inline hlp::byte_offset_t offsetOf(const StridesVar &strides, const IndicesVar &coords)
{
    hlp::byte_offset_t offset(0);
    auto itStrides = strides.cbegin(), endStrides = strides.cend();
    auto itCoords = coords.cbegin(), endCoords = coords.cend();
    for (; itStrides != endStrides && itCoords != endCoords; ++itStrides, ++itCoords)
        offset += *itStrides * *itCoords;
    return offset;
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
hlp::byte_offset_t _indexOf_forward(const strides<_D> &);
template <size_t _D, typename... indices_t>
hlp::byte_offset_t _indexOf_forward(const strides<_D> &strides, size_t coordinate, indices_t... moreCoordinates);

template <size_t _D>
struct strides : std::array<hlp::byte_offset_t, _D> {
	strides()
	{
	}
    strides(const std::array<hlp::byte_offset_t, _D> &other)
        : std::array<hlp::byte_offset_t, _D>(other)
	{
	}
    bool operator==(const std::array<hlp::byte_offset_t, _D> &other)
	{
        return static_cast<std::array<hlp::byte_offset_t, _D> &>(*this) == other;
	}
    bool operator!=(const std::array<hlp::byte_offset_t, _D> &other)
	{
        return static_cast<std::array<hlp::byte_offset_t, _D> &>(*this) != other;
	}

	template <typename... indices_t>
    hlp::byte_offset_t indexOf(size_t coordinate, indices_t... moreCoordinates) const
	{
		static_assert(sizeof...(indices_t) + 1 == _D, "count of indices does not match count of dimensions.");
		return _indexOf_forward(*this, coordinate, moreCoordinates...);
	}

    hlp::byte_offset_t indexOf(std::array<size_t, _D> coordinates) const
	{
        hlp::byte_offset_t pos(0);
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
hlp::byte_offset_t _indexOf_forward(const ndim::strides<_D> &)
{
    return hlp::byte_offset_t(0);
}
template <size_t _D, typename... indices_t>
hlp::byte_offset_t _indexOf_forward(const ndim::strides<_D> &strides, size_t coordinate, indices_t... moreCoordinates)
{
	return strides[_D - 1 - sizeof...(indices_t)] * coordinate + _indexOf_forward(strides, moreCoordinates...);
}

template <size_t _D>
struct layout {
    ndim::sizes<_D> sizes;
    ndim::strides<_D> byte_strides;

    static const size_t Dimensionality = _D;

	layout()
	{
	}
    explicit layout(ndim::sizes<_D> sizes, hlp::byte_offset_t base_stride)
		: sizes(sizes)
    {
		for (size_t d = 0; d < _D; ++d) {
            byte_strides[d] = base_stride;
            base_stride *= sizes[d];
		}
    }
    layout(ndim::sizes<_D> sizes, ndim::strides<_D> byte_strides)
        : sizes(sizes)
        , byte_strides(byte_strides)
    {
    }

	template <typename... coordinatess_t>
    hlp::byte_offset_t offsetOf(size_t coordinate, coordinatess_t... moreCoordinates) const
	{
		static_assert(sizeof...(coordinatess_t) + 1 == _D, "count of coordinates does not match count of dimensions.");
        return byte_strides.indexOf(coordinate, moreCoordinates...);
	}

	template <typename... coordinatess_t>
    hlp::byte_offset_t operator()(size_t coordinate, coordinatess_t... moreCoordinates) const
	{
		static_assert(sizeof...(coordinatess_t) + 1 == _D, "count of coordinates does not match count of dimensions.");
        return byte_strides.indexOf(coordinate, moreCoordinates...);
	}

    hlp::byte_offset_t offsetOf(const std::array<size_t, _D> &coordinates) const
	{
        return byte_strides.indexOf(coordinates);
	}

    hlp::byte_offset_t operator[](const std::array<size_t, _D> &coordinates) const
	{
        return byte_strides.indexOf(coordinates);
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

    size_t count() const
    {
        size_t count = 1;
        for (size_t s : sizes)
            count *= s;
        return count;
    }

	size_t size() const
	{
        return count();
	}
	bool isEmpty() const
	{
		return sizes.isEmpty();
	}
    bool isContiguous(hlp::byte_offset_t base_stride) const
	{

        hlp::byte_offset_t stride = base_stride;
		for (size_t d = 0; d < _D; ++d) {
            if (stride != byte_strides[d])
				return false;
			stride *= sizes[d];
		}
		return true;
	}

	layout<_D - 1> removeDimension(size_t dimension) const
	{
        return layout<_D - 1>(hlp::array::remove(sizes, dimension), hlp::array::remove(byte_strides, dimension));
	}

	layout<_D + 1> addDimension(size_t dimension, size_t virtualSize) const
	{
        return layout<_D + 1>(hlp::array::insert(sizes, dimension, virtualSize), hlp::array::insert(byte_strides, dimension, 0));
	}

	template <size_t _AD>
	layout<_D + _AD> addDimensions(Indices<_AD> dimensions, Sizes<_AD> virtualSizes) const
	{
        return layout<_D + _AD>(
            hlp::array::insert(sizes, virtualSizes, dimensions), hlp::array::insert(byte_strides, dimensions, hlp::byte_offset_t(0)));
	}

	template <size_t _SD>
	layout<_SD> selectDimensions(std::array<size_t, _SD> dimensions) const
	{
        return layout<_SD>(sizes.selectDimensions(dimensions), byte_strides.selectDimensions(dimensions));
	}

    std::array<hlp::byte_offset_t, _D> getHops() const
	{
        std::array<hlp::byte_offset_t, _D> hops;
        hops[0] = byte_strides[0];
		for (size_t d = 1; d < _D; ++d)
            hops[d] = byte_strides[d] - byte_strides[d - 1] * sizes[d - 1];
		return hops;
	}
};

struct LayoutVar {
    ShapeVar shape;
    StridesVar strides;

    size_t dimensionality() const
    {
        return shape.size();
    }

    LayoutVar()
    {
    }
    LayoutVar(ShapeVar shape, hlp::byte_offset_t baseStride)
        : shape(std::move(shape))
    {
        strides.resize(this->shape.size());
        for (size_t i = 0, ii = this->shape.size(); i < ii; ++i) {
            strides[i] = baseStride;
            baseStride *= this->shape[i];
        }
    }
    LayoutVar(ShapeVar shape, StridesVar strides)
        : shape(std::move(shape))
        , strides(std::move(strides))
    {
        if (shape.size() != strides.size())
            throw std::out_of_range("Shape and strides must have same dimensionality.");
    }

    template <size_t Dimensionality>
    LayoutVar(layout<Dimensionality> other)
        : shape(other.sizes.cbegin(), other.sizes.cend())
        , strides(other.byte_strides.cbegin(), other.byte_strides.cend())
    {
    }

    template <size_t Dimensionality>
    layout<Dimensionality> fixDimensionality() const
    {
        if (shape.size() != Dimensionality || strides.size() != Dimensionality)
            throw std::out_of_range("Shape and/or strides must have the fix-dimensionality.");
        layout<Dimensionality> result;
        std::copy(shape.cbegin(), shape.cend(), result.sizes.begin());
        std::copy(strides.cbegin(), strides.cend(), result.byte_strides.begin());
        return result;
    }

    hlp::byte_offset_t offsetOf(const IndicesVar &coords) const
    {
        if (strides.size() != coords.size())
            throw std::out_of_range("Coords and strides must have same dimensionality.");
        hlp::byte_offset_t offset(0);
        for (size_t i = 0, ii = strides.size(); i < ii; ++i)
            offset += coords[i] * strides[i];
        return offset;
    }

    size_t count() const
    {
        size_t count = 1;
        for (size_t s : shape)
            count *= s;
        return count;
    }
};

} // namespace ndim

#endif // NDIM_LAYOUT_H
