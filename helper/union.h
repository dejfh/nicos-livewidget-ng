#ifndef HELPER_UNION_H
#define HELPER_UNION_H

#include <stdexcept>
#include <new>

#include "helper/variadic.h"

namespace hlp
{

template <typename _Type0, typename... _TypeN>
class unionOf
{
public:
	template <size_t N>
	using nthType_t = hlp::variadic::NthType_t<N, _Type0, _TypeN...>;

	unionOf();
	~unionOf();

	template <size_t I>
	nthType_t<I> &get();
	template <size_t I>
	const nthType_t<I> &get() const;

	template <typename _OutType>
	_OutType &getAs();

	void clear();

	template <size_t I, typename... _Args>
	nthType_t<I> &set(_Args &&... args);

protected:
	union {
		struct {
			int m_currentType;
			_Type0 m_value;
		};
		unionOf<_TypeN...> m_next;
	};

	template <size_t I>
	nthType_t<I> &_get();
	template <size_t I>
	const nthType_t<I> &_get() const;

	template <typename _OutType>
	_OutType &_getAs(int i);

	void _clear(int i);
};

template <typename _Type0, typename... _TypeN>
unionOf<_Type0, _TypeN...>::unionOf()
	: m_currentType(-1)
{
}
template <typename _Type0, typename... _TypeN>
unionOf<_Type0, _TypeN...>::~unionOf()
{
	clear();
}

template <typename _Type0, typename... _TypeN>
template <size_t I>
unionOf<_Type0, _TypeN...>::nthType_t<I> &unionOf<_Type0, _TypeN...>::get()
{
	if (I != m_currentType)
		throw std::logic_error("Wrong current type.");
	return this->_get<I>();
}
template <typename _Type0, typename... _TypeN>
template <size_t I>
const unionOf<_Type0, _TypeN...>::nthType_t<I> &unionOf<_Type0, _TypeN...>::get() const
{
	if (I != m_currentType)
		throw std::logic_error("Wrong current type.");
	return this->_get<I>();
}

template <typename _Type0, typename... _TypeN>
template <typename _OutType>
_OutType &unionOf<_Type0, _TypeN...>::getAs()
{
	return this->_getAs<_OutType>(m_currentType);
}

template <typename _Type0, typename... _TypeN>
void unionOf<_Type0, _TypeN...>::clear()
{
	if (m_currentType >= 0)
		_clear(m_currentType);
	m_currentType = -1;
}

template <typename _Type0, typename... _TypeN>
template <size_t I, typename... _Args>
unionOf<_Type0, _TypeN...>::nthType_t<I> &unionOf<_Type0, _TypeN...>::set(_Args &&... args)
{
	if (m_currentType >= 0)
		clear();
	nthType_t<I> &value = _get<I>();
	new (&value) nthType_t<I>(std::forward<_Arg>(args)...);
	m_currentType = I;
	return value;
}

template <typename _Type0, typename... _TypeN>
template <size_t I>
hlp::variadic::NthType_t<I, _Type0, _TypeN...> &unionOf<_Type0, _TypeN...>::_get()
{
	return this->m_next._get<I - 1>();
}
template <typename _Type0, typename... _TypeN>
_Type0 &unionOf<_Type0, _TypeN...>::_get<0>()
{
	return this->m_value;
}

template <typename _Type0, typename... _TypeN>
template <size_t I>
const hlp::variadic::NthType_t<I, _Type0, _TypeN...> &unionOf<_Type0, _TypeN...>::_get() const
{
	return this->m_next._get<I - 1>();
}
template <typename _Type0, typename... _TypeN>
const _Type0 &unionOf<_Type0, _TypeN...>::_get<0>() const
{
	return this->m_value;
}

template <typename _Type0, typename... _TypeN>
template <typename _OutType>
_OutType &unionOf<_Type0, _TypeN...>::_getAs(int i)
{
	if (i == 0)
		return m_value;
	else
		return m_next._getAs<_OutType>();
}

template <typename _Type0, typename... _TypeN>
void unionOf<_Type0, _TypeN...>::_clear(int i)
{
	if (i == 0)
		m_value.~_Type0();
	else
		m_next._clear(i - 1);
}

template <typename _Type>
class unionOf<_Type>
{
	int m_currentType;

protected:
	_Type m_currentValue;
};

} // namespace hlp

#endif // HELPER_UNION_H
