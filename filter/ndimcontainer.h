#ifndef FILTER_NDIMCONTAINER_H
#define FILTER_NDIMCONTAINER_H

#include "ndim/pointer.h"

namespace filter
{

template <typename _ElementType, size_t _Dimensionality = 0>
class Container
{
public:
	using ElementType = _ElementType;
	using MutableElementType = typename std::remove_cv<ElementType>::type;
	static const size_t Dimensionality = _Dimensionality;

	friend class Container<MutableElementType, Dimensionality>;
	friend class Container<const ElementType, Dimensionality>;

	std::vector<MutableElementType> m_buffer;
	ndim::pointer<ElementType, Dimensionality> m_pointer;

public:
	Container()
	{
	}
	explicit Container(ndim::pointer<_ElementType, Dimensionality> pointer)
		: m_pointer(pointer)
	{
	}
	explicit Container(ndim::sizes<Dimensionality> sizes)
		: m_buffer(sizes.size())
		, m_pointer(m_buffer.data(), sizes)
	{
	}

	Container(const Container<ElementType, Dimensionality> &other) = delete;
	Container<ElementType, Dimensionality> &operator=(const Container<ElementType, Dimensionality> &other) = delete;

	template <typename _OtherType>
	Container(Container<_OtherType, Dimensionality> &&other)
		: m_buffer(std::move(other.m_buffer))
		, m_pointer(std::move(other.m_pointer))
	{
	}
	template <typename _OtherType>
	Container<ElementType, Dimensionality> &operator=(Container<_OtherType, Dimensionality> &&other)
	{
		m_buffer = std::move(other.m_buffer);
		m_pointer = std::move(other.m_pointer);
		return *this;
	}

	void reset()
	{
		m_buffer.clear();
	}
	void reset(ndim::pointer<ElementType, Dimensionality> pointer)
	{
		m_buffer.clear();
		m_pointer = pointer;
	}
	ndim::pointer<MutableElementType, Dimensionality> reset(ndim::sizes<Dimensionality> sizes)
	{
		m_buffer.resize(sizes.size());
		m_pointer = ndim::pointer<ElementType, Dimensionality>(m_buffer.data(), sizes);
		return ndim::pointer<MutableElementType, Dimensionality>(m_buffer.data(), m_pointer.getLayout());
	}
	void resize(ndim::sizes<Dimensionality> sizes = ndim::sizes<0>())
	{
		if (m_pointer.data && m_pointer.sizes == sizes)
			return;
		reset(sizes);
	}

	const ndim::pointer<ElementType, Dimensionality> &pointer() const
	{
		return m_pointer;
	}
	ndim::pointer<MutableElementType, Dimensionality> mutablePointer()
	{
		if (m_buffer.empty())
			throw std::logic_error("Can not get mutable pointer, without ownership.");
		return ndim::pointer<MutableElementType, Dimensionality>(m_buffer.data(), m_pointer.getLayout());
	}

	bool ownsData() const
	{
		return !m_buffer.empty();
	}
	bool hasData() const
	{
		return m_pointer.data != nullptr;
	}
};

} // namespace filter

#endif // FILTER_NDIMCONTAINER_H
