#ifndef NDIM_MULTIITERATOR_H
#define NDIM_MULTIITERATOR_H

#include <ndim/pointer.h>
#include <tuple>

#include <iterator>
#include <cstddef>

#include "variadic.h"

namespace ndim
{

template <size_t _D, typename... _Ts>
class multiIterator
{
	std::tuple<_Ts *...> m_pointers;
	std::array<std::array<ptrdiff_t, _D>, sizeof...(_Ts)> m_hops;
	std::array<size_t, _D> m_pos;
	std::array<size_t, _D> m_sizes;

	std::array<ptrdiff_t, _D> _makeHops(std::array<ptrdiff_t, _D> strides, std::array<size_t, _D> sizes)
	{
		std::array<ptrdiff_t, _D> hops;
		hops[0] = strides[0];
		for (size_t d = 1; d < _D; ++d)
			hops[d] = strides[d] - sizes[d - 1] * strides[d - 1];
		return hops;
	}

	template <size_t I = sizeof...(_Ts)-1>
	void _init(std::tuple<ndim::pointer<_Ts, _D>...> pointers)
	{
		if (I > 0)
			_init<(I ? I - 1 : 0)>(pointers);
		else
			m_sizes = std::get<0>(pointers).sizes;
		std::get<I>(m_pointers) = std::get<I>(pointers).data;
		std::get<I>(m_hops) = _makeHops(std::get<I>(pointers).strides, std::get<I>(pointers).sizes);
		assert(std::get<I>(pointers).sizes == m_sizes);
	}

	template <size_t I = sizeof...(_Ts)-1>
	void _hop(size_t d, ptrdiff_t delta)
	{
		if (I > 0)
			_hop<(I ? I - 1 : 0)>(d, delta);
		std::get<I>(m_pointers) += std::get<I>(m_hops)[d] * delta;
	}

	template <size_t... I>
	std::tuple<_Ts &...> _deref(jfh::variadic::Sequence<I...>) const
	{
		return std::tuple<_Ts &...>(*std::get<I>(m_pointers)...);
	}

public:
	multiIterator(std::tuple<ndim::pointer<_Ts, _D>...> pointers)
	{
		m_pos.fill(0);
		_init(pointers);
	}

	multiIterator(ndim::pointer<_Ts, _D>... pointers)
	{
		m_pos.fill(0);
		_init(std::tuple<ndim::pointer<_Ts, _D>...>(pointers...));
	}

	std::array<size_t, _D> coords() const
	{
		return m_pos;
	}

	std::tuple<_Ts &...> operator*() const
	{
		return _deref(jfh::variadic::makeSequence(m_pointers));
	}

	multiIterator<_D, _Ts...> &operator++()
	{
		for (size_t d = 0; d < _D - 1; ++d) {
			_hop(d, 1);
			if (++this->m_pos[d] != this->m_sizes[d])
				return *this;
			this->m_pos[d] = 0;
		}
		_hop(_D - 1, 1);
		++this->m_pos[_D - 1];
		return *this;
	}
	multiIterator<_D, _Ts...> &operator--()
	{
		for (size_t d = 0; d < _D - 1; --d) {
			_hop(d, -1);
			if (this->m_pos[d]-- != 0)
				return *this;
			this->m_pos[d] = this->m_sizes[d] - 1;
		}
		_hop(_D - 1, -1);
		--this->m_pos[_D - 1];
		return *this;
	}

	multiIterator<_D, _Ts...> operator++(int)
	{
		multiIterator<_D, _Ts...> other(*this);
		++(*this);
		return other;
	}
	multiIterator<_D, _Ts...> operator--(int)
	{
		multiIterator<_D, _Ts...> other(*this);
		--(*this);
		return other;
	}

	multiIterator<_D, _Ts...> &operator+=(ptrdiff_t delta)
	{
		if (delta == 0)
			return *this;
		if (delta < 0)
			return (*this) -= -delta;
		for (size_t d = 0; d < _D - 1; ++d) {
			_hop(d, delta);
			size_t toNextHop = this->m_sizes[d] - this->m_pos[d];
			size_t remainder = delta % this->m_sizes[d];
			delta /= this->m_sizes[d];
			if (remainder < toNextHop)
				this->m_pos[d] += remainder;
			else {
				++delta;
				this->m_pos[d] = remainder - toNextHop;
			}
		}
		_hop(_D - 1, delta);
		this->m_pos[_D - 1] += delta;
		return *this;
	}
	multiIterator<_D, _Ts...> &operator-=(ptrdiff_t delta)
	{
		if (delta == 0)
			return *this;
		if (delta < 0)
			return (*this) += -delta;
		for (size_t d = 0; d < _D - 1; ++d) {
			_hop(d, -delta);
			size_t toNextHop = this->m_pos[d] + 1;
			size_t remainder = delta % this->m_sizes[d];
			delta /= this->m_sizes[d];
			if (remainder < toNextHop)
				this->m_pos[d] -= remainder;
			else {
				++delta;
				this->m_pos[d] = this->m_sizes[d] - 1 - (remainder - toNextHop);
			}
		}
		_hop(_D - 1, -delta);
		this->m_pos[_D - 1] -= delta;
		return *this;
	}

	multiIterator<_D, _Ts...> operator+(ptrdiff_t delta) const
	{
		multiIterator<_D, _Ts...> it(*this);
		return it += delta;
	}
	multiIterator<_D, _Ts...> operator-(ptrdiff_t delta) const
	{
		multiIterator<_D, _Ts...> it(*this);
		return it -= delta;
	}

	ptrdiff_t operator-(const multiIterator<_D, _Ts...> &other) const
	{
		ptrdiff_t delta = 0;
		size_t factor = 1;
		for (size_t d = 0; d < _D; ++d) {
			delta += (other.m_pos[d] - this->m_pos[d]) * factor;
			factor += this->m_sizes[d];
		}
		return delta;
	}

	bool operator<(const multiIterator<_D, _Ts...> &other) const
	{
		for (size_t d = _D - 1; d >= 0; --d)
			if (this->m_pos[d] < other.m_pos[d])
				return true;
		return false;
	}
	bool operator>(const multiIterator<_D, _Ts...> &other) const
	{
		for (size_t d = _D - 1; d >= 0; --d)
			if (this->m_pos[d] > other.m_pos[d])
				return true;
		return false;
	}
	bool operator<=(const multiIterator<_D, _Ts...> &other) const
	{
		for (size_t d = _D - 1; d >= 0; --d)
			if (this->m_pos[d] <= other.m_pos[d])
				return true;
		return false;
	}
	bool operator>=(const multiIterator<_D, _Ts...> &other) const
	{
		for (size_t d = _D - 1; d >= 0; --d)
			if (this->m_pos[d] >= other.m_pos[d])
				return true;
		return false;
	}

	bool operator==(const multiIterator<_D, _Ts...> &other) const
	{
		return this->m_pos == other.m_pos;
	}
	bool operator!=(const multiIterator<_D, _Ts...> &other) const
	{
		return this->m_pos != other.m_pos;
	}
};

template <typename _T, size_t _D>
class iterator : public multiIterator<_D, _T>, public std::iterator<std::random_access_iterator_tag, _T>
{
public:
	iterator(ndim::pointer<_T, _D> pointer)
		: multiIterator<_D, _T>(pointer)
	{
	}

	_T &operator*() const
	{
		return std::get<0>(multiIterator<_D, _T>::operator*());
	}
	_T *operator->() const
	{
		return &std::get<0>(multiIterator<_D, _T>::operator*());
	}

	iterator<_T, _D> &operator++()
	{
		multiIterator<_D, _T>::operator++();
		return *this;
	}
	iterator<_T, _D> &operator--()
	{
		multiIterator<_D, _T>::operator--();
		return *this;
	}

	iterator<_T, _D> operator++(int)
	{
		iterator<_T, _D> other(*this);
		++(*this);
		return other;
	}
	iterator<_T, _D> operator--(int)
	{
		iterator<_T, _D> other(*this);
		--(*this);
		return other;
	}

	iterator<_T, _D> &operator+=(ptrdiff_t delta)
	{
		multiIterator<_D, _T>::operator+=(delta);
		return *this;
	}
	iterator<_T, _D> &operator-=(ptrdiff_t delta)
	{
		multiIterator<_D, _T>::operator-=(delta);
		return *this;
	}

	iterator<_T, _D> operator+(ptrdiff_t delta) const
	{
		iterator<_T, _D> it(*this);
		return it += delta;
	}
	iterator<_T, _D> operator-(ptrdiff_t delta) const
	{
		iterator<_T, _D> it(*this);
		return it -= delta;
	}

	ptrdiff_t operator-(const iterator<_T, _D> &other) const
	{
		return multiIterator<_D, _T>::operator-(other);
	}

	//	bool operator<(const iterator<_T, _D> &other) const
	//	{
	//		return multiIterator<_D, _T>::operator<(other);
	//	}
	//	bool operator>(const iterator<_T, _D> &other) const
	//	{
	//		return multiIterator<_D, _T>::operator>(other);
	//	}
	//	bool operator<=(const iterator<_T, _D> &other) const
	//	{
	//		return multiIterator<_D, _T>::operator<=(other);
	//	}
	//	bool operator>=(const iterator<_T, _D> &other) const
	//	{
	//		return multiIterator<_D, _T>::operator>=(other);
	//	}

	//	bool operator==(const iterator<_T, _D> &other) const
	//	{
	//		return multiIterator<_D, _T>::operator==(other);
	//	}
	//	bool operator!=(const iterator<_T, _D> &other) const
	//	{
	//		return multiIterator<_D, _T>::operator!=(other);
	//	}
};

template <size_t _Dimensions, typename... _Types>
multiIterator<_Dimensions, _Types...> makeMultiIterator(std::tuple<ndim::pointer<_Types, _Dimensions>...> pointers)
{
	return multiIterator<_Dimensions, _Types...>(pointers);
}

template <size_t _Dimensions, typename _Type>
iterator<_Type, _Dimensions> makeIterator(ndim::pointer<_Type, _Dimensions> pointer)
{
	return iterator<_Type, _Dimensions>(pointer);
}

} // namespace ndim

#endif // NDIM_MULTIITERATOR_H
