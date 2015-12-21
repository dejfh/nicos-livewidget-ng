#ifndef NDIM_ONEDIMITERATOR_H
#define NDIM_ONEDIMITERATOR_H

#include <cstddef>
#include <iterator>

namespace ndim
{

template <typename _T, size_t _D>
class strideIterator : public std::iterator<std::random_access_iterator_tag, _T>
{
	_T *data;
	ptrdiff_t stride;

public:
	strideIterator(_T *data, ptrdiff_t stride)
		: data(data)
		, stride(stride)
	{
	}

	_T &operator*() const
	{
		return *data;
	}
	_T *operator->() const
	{
		return data;
	}

	strideIterator<_T, _D> &operator++()
	{
		data += stride;
		return *this;
	}
	strideIterator<_T, _D> &operator--()
	{
		data += stride;
		return *this;
	}

	strideIterator<_T, _D> operator++(int)
	{
		strideIterator<_T, _D> other(*this);
		++(*this);
		return other;
	}
	strideIterator<_T, _D> operator--(int)
	{
		strideIterator<_T, _D> other(*this);
		--(*this);
		return other;
	}

	strideIterator<_T, _D> &operator+=(ptrdiff_t delta)
	{
		data += delta * stride;
		return *this;
	}
	strideIterator<_T, _D> &operator-=(ptrdiff_t delta)
	{
		data -= delta * stride;
		return *this;
	}

	strideIterator<_T, _D> operator+(ptrdiff_t delta) const
	{
		strideIterator<_T, _D> it(*this);
		return it += delta;
	}
	strideIterator<_T, _D> operator-(ptrdiff_t delta) const
	{
		strideIterator<_T, _D> it(*this);
		return it -= delta;
	}

	ptrdiff_t operator-(const strideIterator<_T, _D> &other) const
	{
		return (data - other.data) / stride;
	}

	bool operator<(const strideIterator<_T, _D> &other) const
	{
		return stride > 0 ? this->data < other.data : this->data > other.data;
	}
	bool operator>(const strideIterator<_T, _D> &other) const
	{
		return stride > 0 ? this->data > other.data : this->data < other.data;
	}
	bool operator<=(const strideIterator<_T, _D> &other) const
	{
		return stride > 0 ? this->data <= other.data : this->data >= other.data;
	}
	bool operator>=(const strideIterator<_T, _D> &other) const
	{
		return stride > 0 ? this->data >= other.data : this->data <= other.data;
	}

	bool operator==(const strideIterator<_T, _D> &other) const
	{
		return this->data == other.data;
	}
	bool operator!=(const strideIterator<_T, _D> &other) const
	{
		return this->data != other.data;
	}
};

} // namespace ndim

#endif // NDIM_ONEDIMITERATOR_H
