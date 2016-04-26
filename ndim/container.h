#ifndef NDIM_CONTAINER_H
#define NDIM_CONTAINER_H

#include <vector>

#include "ndim/pointer.h"

namespace ndim
{

template <typename _ElementType>
class ContainerBase
{
public:
    using ElementType = _ElementType;

protected:
    std::vector<ElementType> m_vector;
    ElementType *m_mutableData;
    const ElementType *m_constData;

public:
    ContainerBase()
        : m_mutableData(nullptr)
        , m_constData(nullptr)
    { }
    ~ContainerBase() = default;
    ContainerBase(const ContainerBase<ElementType> &other) = delete;
    ContainerBase(ContainerBase<ElementType> &&other) = default;

    size_t capacity() const
    {
        return m_vector.capacity();
    }
    bool ownsData() const
    {
        return !m_vector.empty();
    }

    void swapOwnership(ContainerBase<_ElementType> &other)
    {
        std::swap(this->m_vector, other.m_vector);
    }
};

template <typename _ElementType>
class ContainerVar : public ContainerBase<_ElementType>
{
public:
    using ElementType = _ElementType;

private:
    ndim::LayoutVar m_layout;
};

template <typename _ElementType, size_t _Dimensionality = 0>
class Container : public ContainerBase<_ElementType>
{
public:
    using ElementType = _ElementType;
    static const size_t Dimensionality = _Dimensionality;

private:
    ndim::layout<Dimensionality> m_layout;

public:
    Container()
        : m_mutableData(nullptr)
        , m_constData(nullptr)
    {
    }

    explicit Container(ContainerBase<ElementType> &&recycle)
        : ContainerBase<ElementType>(std::move(recycle))
        , m_mutableData(nullptr)
        , m_constData(nullptr)
    {
    }

    Container(const Container<ElementType, Dimensionality> &other) = delete;

    Container(Container<ElementType, Dimensionality> &&other)
        : ContainerBase<ElementType>(std::move(other))
        , m_mutableData(other.m_mutableData)
        , m_constData(other.m_constData)
        , m_layout(other.m_layout)
    {
        other.m_mutableData = nullptr;
        other.m_constData = nullptr;
    }

    Container(ndim::pointer<ElementType, Dimensionality> data)
        : m_mutableData(data.data)
        , m_constData(data.data)
        , m_layout(data.getLayout())
    {
    }

    Container(ndim::pointer<const ElementType, Dimensionality> data)
        : m_mutableData(nullptr)
        , m_constData(data.data)
        , m_layout(data.getLayout())
    {
    }

    Container<ElementType, Dimensionality> &operator=(const Container<ElementType, Dimensionality> &other) = delete;

    Container<ElementType, Dimensionality> &operator=(Container<ElementType, Dimensionality> &&other)
    {
        this->m_vector = std::move(other.m_vector);
        this->m_mutableData = other.m_mutableData;
        this->m_constData = other.m_constData;
        this->m_layout = other.m_layout;
        other.m_mutableData = nullptr;
        other.m_constData = nullptr;

        return *this;
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

    void resize(ndim::sizes<Dimensionality> sizes)
    {
        this->m_vector.resize(sizes.size());
        m_constData = m_mutableData = this->m_vector.data();
        m_layout = ndim::layout<Dimensionality>(sizes, hlp::byte_offset_t::inArray<ElementType>());
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
};

template <typename ElementType>
Container<ElementType> makeRefContainer(ElementType &element)
{
    return ndim::make_pointer(element);
}

template <typename ElementType>
Container<ElementType> makeRefContainer(const ElementType &element)
{
    return ndim::make_pointer(element);
}

template <typename ElementType>
Container<ElementType> makeConstRefContainer(const ElementType &element)
{
    return ndim::make_pointer(element);
}

template <typename ElementType, size_t Dimensionality>
Container<ElementType, Dimensionality> makeConstRefContainer(ndim::pointer<const ElementType, Dimensionality> data)
{
    return data;
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

} // namespace ndim

#endif // NDIM_CONTAINER_H
