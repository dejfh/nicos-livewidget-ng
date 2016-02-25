#ifndef FILTER_NDIMCONTAINER_H
#define FILTER_NDIMCONTAINER_H

#include <vector>

#include "ndim/pointer.h"

namespace fc
{

template <typename _ElementType, size_t _Dimensionality = 0>
class Container
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	std::vector<ElementType> m_ownership;
	ElementType *m_mutableData;
	const ElementType *m_constData;
	ndim::layout<Dimensionality> m_layout;

public:
	Container()
		: m_mutableData(nullptr)
		, m_constData(nullptr)
	{
	}

	explicit Container(std::vector<ElementType> &&ownership)
		: m_ownership(std::move(ownership))
		, m_mutableData(nullptr)
		, m_constData(nullptr)
	{
	}

	Container(const Container<ElementType, Dimensionality> &other) = delete;
	Container(Container<ElementType, Dimensionality> &&other)
		: m_ownership(std::move(other.m_ownership))
		, m_mutableData(other.m_mutableData)
		, m_constData(other.m_constData)
		, m_layout(other.m_layout)
	{
		other.m_mutableData = nullptr;
		other.m_constData = nullptr;
	}

	template <size_t OtherDimensionality>
	explicit Container(const Container<ElementType, OtherDimensionality> &&other)
		: m_ownership(std::move(other.m_ownership))
		, m_mutableData(nullptr)
		, m_constData(nullptr)
	{
	}

	Container<ElementType, Dimensionality> &operator=(const Container<ElementType, Dimensionality> &other) = delete;
	Container<ElementType, Dimensionality> &operator=(Container<ElementType, Dimensionality> &&other)
	{
		this->m_ownership = std::move(other.m_ownership);
		this->m_mutableData = other.m_mutableData;
		this->m_constData = other.m_constData;
		this->m_layout = other.m_layout;
		other.m_mutableData = nullptr;
		other.m_constData = nullptr;

		return *this;
	}

	void setMutablePointer(ndim::pointer<ElementType, Dimensionality> data)
	{
		m_ownership = std::move(std::vector<ElementType>());
		m_constData = m_mutableData = data.data;
		m_layout = data.getLayout();
	}

	void setConstPointer(ndim::pointer<const ElementType, Dimensionality> data)
	{
		m_ownership = std::move(std::vector<ElementType>());
		m_mutableData = nullptr;
		m_constData = data.data;
		m_layout = data.getLayout();
	}

	void changePointer(ndim::pointer<ElementType, Dimensionality> data)
	{
		m_constData = m_mutableData = data.data;
		m_layout = data.getLayout();
	}
	void changePointer(ndim::pointer<const ElementType, Dimensionality> data)
	{
		m_mutableData = nullptr;
		m_constData = data.data;
		m_layout = data.getLayout();
	}

	size_t capacity() const
	{
		return m_ownership.capacity();
	}
	void resize(ndim::sizes<Dimensionality> sizes)
	{
		m_ownership.resize(sizes.size());
		m_constData = m_mutableData = m_ownership.data();
		m_layout = ndim::layout<Dimensionality>(sizes, hlp::byte_offset_t::inArray<ElementType>());
	}

	bool ownsData() const
	{
		return !m_ownership.empty();
	}
	bool hasData() const
	{
		return m_constData;
	}
	bool isMutable() const
	{
		return m_mutableData;
	}
	ndim::pointer<ElementType, Dimensionality> mutableData() const
	{
		return ndim::make_pointer(m_mutableData, m_layout);
	}
	ndim::pointer<const ElementType, Dimensionality> constData() const
	{
		return ndim::make_pointer(m_constData, m_layout);
	}
	const ndim::layout<Dimensionality> layout() const
	{
		return m_layout;
	}

	std::vector<ElementType> takeOwnership()
	{
		return std::move(m_ownership);
	}
	void setOwnership(std::vector<ElementType> &&ownership)
	{
		this->m_ownership = std::move(ownership);
	}
	template <size_t OtherDimensionality>
	void setOwnership(Container<ElementType, OtherDimensionality> &&container)
	{
		this->setOwnership(container.takeOwnership());
	}
};

template <typename ElementType>
Container<ElementType, 0> makeConstRefContainer(const ElementType &element)
{
	Container<ElementType, 0> container;
	container.setConstPointer(ndim::make_pointer(&element, ndim::sizes<0>()));
	return container;
}

template <typename ElementType, size_t Dimensionality>
Container<ElementType, Dimensionality> makeConstRefContainer(ndim::pointer<const ElementType, Dimensionality> data)
{
	Container<ElementType, Dimensionality> container;
	container.setConstPointer(data);
	return container;
}

template <typename ElementType>
Container<ElementType, 0> makeMutableRefContainer(ElementType &element)
{
	Container<ElementType, 0> container;
	container.setMutablePointer(ndim::make_pointer(&element, ndim::sizes<0>()));
	return container;
}

template <typename ElementType, size_t Dimensionality>
Container<ElementType, Dimensionality> makeMutableContainer(std::array<size_t, Dimensionality> sizes, Container<ElementType, Dimensionality> *recycle)
{
	ndim::pointer<ElementType, Dimensionality> data = recycle->mutableData();
	if (data.data && (data.sizes == sizes))
		return std::move(*recycle);
	if (recycle->capacity() >= ndim::totalCount<Dimensionality>(sizes)) {
		recycle->resize(sizes);
		return std::move(*recycle);
	}
	Container<ElementType, Dimensionality> container;
	container.resize(sizes);
	return container;
}

template <typename ElementType>
Container<ElementType> makeMutableContainer(Container<ElementType> *recycle)
{
	return makeMutableContainer(ndim::Sizes<0>(), recycle);
}

} // namespace fc

#endif // FILTER_NDIMCONTAINER_H
