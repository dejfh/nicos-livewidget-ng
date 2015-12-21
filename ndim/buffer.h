#ifndef NDIM_VECTOR_H
#define NDIM_VECTOR_H

#include <vector>
#include <numeric>

#include <ndim/pointer.h>
#include <ndim/layout.h>
#include <ndim/algorithm.h>

namespace ndim
{

template <typename _T, size_t _D>
class Buffer
{
	ndim::sizes<_D> m_sizes;
	std::vector<_T> m_data;

public:
	size_t size() const
	{
		return m_data.size();
	}
	const ndim::sizes<_D> &sizes() const
	{
		return m_sizes;
	}
	const _T *data() const
	{
		return m_data.data();
	}
	_T *data()
	{
		return m_data.data();
	}

	ndim::pointer<_T, _D> pointer()
	{
		return ndim::pointer<_T, _D>(m_data.data(), m_sizes);
	}
	ndim::pointer<const _T, _D> cpointer() const
	{
		return ndim::pointer<const _T, _D>(m_data.data(), m_sizes);
	}
	ndim::layout<_D> layout() const
	{
		return ndim::layout<_D>(m_sizes);
	}

	operator ndim::pointer<_T, _D>()
	{
		return pointer();
	}
	operator ndim::pointer<const _T, _D>()
	{
		return pointer();
	}

	operator const ndim::layout<_D>()
	{
		return layout();
	}

	Buffer()
	{
		m_sizes.fill(0);
	}

	explicit Buffer(std::vector<_T> &&data, std::array<size_t, _D> sizes)
		: m_sizes(sizes)
		, m_data(std::move(data))
	{
	}

	explicit Buffer(ndim::sizes<_D> sizes)
		: m_sizes(sizes)
		, m_data(sizes.size())
	{
	}

	explicit Buffer(ndim::pointer<_T, _D> data)
		: m_sizes(data.sizes)
	{
		m_data.resize(data.size());
		ndim::copy(data, *this);
	}

	void resize(ndim::sizes<_D> sizes)
	{
		m_data.resize(sizes.size());
		m_sizes = sizes;
	}

	Buffer<_T, _D> &operator=(ndim::pointer<_T, _D> data)
	{
		m_data.resize(data.size());
		m_sizes = data.sizes;
		ndim::copy(data, *this);
		return *this;
	}

	template <typename _T_other, typename = typename std::enable_if<std::is_same<typename std::remove_cv<_T>::type, _T_other>::value>::type>
	explicit Buffer(ndim::pointer<_T_other, _D> data)
		: m_sizes(data.sizes)
	{
		m_data.resize(data.size());
		ndim::copy(data, *this);
	}

	template <typename _T_other, typename = typename std::enable_if<std::is_same<typename std::remove_cv<_T>::type, _T_other>::value>::type>
	Buffer<_T, _D> &operator=(ndim::pointer<_T_other, _D> data)
	{
		m_data.resize(data.size());
		m_sizes = data.sizes;
		ndim::copy(data, *this);
		return *this;
	}
};

} // namespace ndim

#endif // NDIM_VECTOR_H
