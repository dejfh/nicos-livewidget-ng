#ifndef NDIM_POINTER_H
#define NDIM_POINTER_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

#include "helper/helper.h"
#include "helper/byteoffset.h"

#include "ndim/layout.h"
#include "ndim/range.h"

namespace ndim
{

template <typename _T, size_t _D>
struct pointer : ::ndim::layout<_D> {
    using ElementType = _T;

	_T *data;

	_T &first() const
	{
		return *data;
	}
	_T &last() const
	{
		_T *item = data;
		for (size_t i = 0; i < _D; ++i)
			item += (this->sizes[i] - 1) * this->strides[i];
		return *item;
	}

	pointer()
		: data(nullptr)
	{
	}

    pointer(_T *data, const ::ndim::layout<_D> &layout)
        : ::ndim::layout<_D>(layout)
        , data(data)
    {
    }

    pointer(_T *data, const std::array<size_t, _D> &shape, const std::array<hlp::byte_offset_t, _D> &byteStrides)
        : layout<_D>(shape, byteStrides)
        , data(data)
    {
    }

    pointer(_T *data, ndim::sizes<_D> shape)
        : layout<_D>(shape, hlp::byte_offset_t::inArray<_T>())
		, data(data)
	{
	}

    template <typename OtherElementType, typename = typename std::enable_if<std::is_assignable<ElementType *&, OtherElementType *>::value>::type>
    pointer(const ndim::pointer<OtherElementType, _D> &other)
		: layout<_D>(other)
		, data(other.data)
	{
	}

    template <typename OtherElementType, typename = typename std::enable_if<std::is_assignable<ElementType *&, OtherElementType *>::value>::type>
    pointer<_T, _D> &operator=(const pointer<OtherElementType, _D> &other)
	{
		data = other.data;
		static_cast<ndim::layout<_D> &>(*this) = static_cast<const ndim::layout<_D> &>(other);
		return *this;
	}

	template <typename... coords_t>
	_T &operator()(size_t coordinate, coords_t... coordinates) const
	{
        return *(data + this->offsetOf(coordinate, coordinates...));
	}

	_T &operator[](ndim::Indices<_D> coordinates) //
	{
        return *(data + this->offsetOf(coordinates));
	}

	_T value(ndim::Indices<_D> coordinates, const _T &_default = _T())
	{
		if (this->sizes.contains(coordinates))
			return this->operator[](coordinates);
		return _default;
	}

	template <typename... coords_t>
	_T value(const _T &_default, size_t coordinate, coords_t... coordinates)
	{
		std::array<size_t, _D> coords = {coordinate, size_t(coordinates)...};
		return value(coords, _default);
	}

	operator bool() const
	{
		return data;
	}

    bool isContiguous() const
    {
        return ndim::layout<_D>::isContiguous(hlp::byte_offset_t::inArray<_T>());
    }

	const ndim::layout<_D> &getLayout() const
	{
		return *this;
	}

	ndim::layout<_D> &getLayout()
	{
		return *this;
	}

	pointer<_T, _D - 1> removeDimension(size_t dimension, size_t index) const
	{
        _T *newData = data + this->byte_strides[dimension] * index;
		return pointer<_T, _D - 1>(newData, this->getLayout().removeDimension(dimension));
	}
	pointer<_T, _D + 1> addDimension(size_t dimension, size_t virtual_size) const
	{
		return pointer<_T, _D + 1>(data, getLayout().addDimension(dimension, virtual_size));
	}

	template <size_t _AD>
	pointer<_T, _D + _AD> addDimensions(Indices<_AD> dimensions, ndim::sizes<_AD> virtualSizes)
	{
		return pointer<_T, _D + _AD>(data, getLayout().addDimensions(dimensions, virtualSizes));
	}
	template <size_t _AD>
	pointer<_T, _D + _AD> addDimensions(Indices<_AD> dimensions, Sizes<_AD> virtualSizes)
	{
		return pointer<_T, _D + _AD>(data, getLayout().addDimensions(dimensions, virtualSizes));
	}

	template <size_t _D_select>
	pointer<_T, _D_select> selectDimensions(std::array<size_t, _D_select> dimensions)
	{
		return pointer<_T, _D_select>(data, getLayout().selectDimensions(dimensions));
	}

	void selectRange(size_t dimension, size_t start, size_t end)
	{
		assert(dimension < _D && start <= this->sizes[dimension] && end <= this->sizes[dimension] && start <= end);
        data += start * this->byte_strides[dimension];
		this->sizes[dimension] = end - start;
	}

	void selectRange(::ndim::range<_D> range)
	{
		for (size_t d = 0; d < _D; ++d)
			if (range.coords[d] + range.sizes[d] > this->sizes[d])
				throw std::range_error("selected range exceeds data.");
		this->data = &(*this)[range.coords];
		this->sizes = range.sizes;
	}

	pointer<_T, _D> selectedRange(size_t dimension, size_t start, size_t end) const
	{
		pointer<_T, _D> ptr(*this);
		ptr.selectRange(dimension, start, end);
		return ptr;
	}

	pointer<_T, _D> selectedRange(::ndim::range<_D> range)
	{
		pointer<_T, _D> ptr(*this);
		ptr.selectRange(range);
		return ptr;
	}

	void mirror(size_t dimension)
	{
		assert(dimension < _D);
        data += (this->sizes[dimension] - 1) * this->byte_strides[dimension];
        this->byte_strides[dimension] = -this->byte_strides[dimension];
	}

	pointer<_T, _D> mirrored(size_t dimension)
	{
		pointer<_T, _D> ptr(*this);
		ptr.mirror(dimension);
		return ptr;
	}
};

template <typename _ElementType>
struct PointerVar : LayoutVar {
    using ElementType = _ElementType;

    ElementType *data;

    ElementType &first() const
    {
        return *data;
    }
    ElementType &last() const
    {
        ElementType *item = data;
        for (size_t i = 0, ii = this->dimensionality(); i < ii; ++i)
            item += (this->shape[i] - 1) * this->strides[i];
        return *item;
    }

    PointerVar()
        : data(nullptr)
    {
    }

    PointerVar(ElementType *data, LayoutVar layout)
        : LayoutVar(std::move(layout))
        , data(data)
    {
    }

    PointerVar(ElementType *data, ShapeVar shape, StridesVar byteStrides)
        : LayoutVar(std::move(shape), std::move(byteStrides))
        , data(data)
    {
    }

    PointerVar(ElementType *data, ShapeVar shape)
        : LayoutVar(std::move(shape), hlp::byte_offset_t::inArray<ElementType>())
        , data(data)
    {
    }

    template <size_t Dimensionality>
    PointerVar(pointer<ElementType, Dimensionality> other)
        : LayoutVar(other)
        , data(other.data)
    {
    }

    template <size_t Dimensionality>
    pointer<ElementType, Dimensionality> fixDimensionality() const
    {
        auto layout = this->layout().template fixDimensionality<Dimensionality>();
        return pointer<ElementType, Dimensionality>(data, layout);
    }

    template <typename OtherElementType, typename = typename std::enable_if<std::is_assignable<ElementType *&, OtherElementType *>::value>::type>
    PointerVar(const PointerVar<OtherElementType> &other)
        : LayoutVar(other)
        , data(other.data)
    {
    }

    template <typename OtherElementType, typename = typename std::enable_if<std::is_assignable<ElementType *&, OtherElementType *>::value>::type>
    PointerVar<ElementType> &operator=(PointerVar<OtherElementType> other)
    {
        data = other.data;
        static_cast<LayoutVar &>(*this) = static_cast<LayoutVar &&>(other);
        return *this;
    }

    ElementType &operator[](const IndicesVar &coords) const
    {
        return *(data + this->offsetOf(coords));
    }

    ElementType value(const IndicesVar &coords, ElementType _default) const
    {
        if (ndim::contains(this->shape, coords))
            return this->operator[](coords);
        return _default;
    }

    operator bool() const
    {
        return data;
    }

    const LayoutVar &layout() const
    {
        return *this;
    }

    LayoutVar &layout()
    {
        return *this;
    }
};

template <typename ElementType>
PointerVar<ElementType> make_pointer(ElementType *data, LayoutVar layout)
{
    return PointerVar<ElementType>(data, std::move(layout));
}

template <typename ElementType>
PointerVar<ElementType> make_pointer(ElementType *data, ShapeVar shape)
{
    return PointerVar<ElementType>(data, std::move(shape));
}

template <typename ElementType, size_t Dimensionality>
pointer<ElementType, Dimensionality> make_pointer(ElementType *data, const ndim::layout<Dimensionality> &layout)
{
	return pointer<ElementType, Dimensionality>(data, layout);
}

template <typename ElementType, size_t Dimensionality>
pointer<ElementType, Dimensionality> make_pointer(ElementType *data, const Sizes<Dimensionality> &shape)
{
    return pointer<ElementType, Dimensionality>(data, shape);
}

template <typename ElementType>
pointer<ElementType, 0> make_pointer(ElementType &data)
{
    return pointer<ElementType, 0>(&data, sizes<0>());
}

template <typename _T, typename... SizesTypes>
pointer<_T, 1 + sizeof...(SizesTypes)> make_ptr_contiguous(_T *data, size_t size0, SizesTypes... sizeN)
{
	return pointer<_T, 1 + sizeof...(SizesTypes)>(data, Sizes<1 + sizeof...(SizesTypes)>{size0, sizeN...});
}

} // namespace ndim

#endif // NDIM_POINTER_H
