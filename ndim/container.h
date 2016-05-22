#ifndef NDIM_CONTAINER_H
#define NDIM_CONTAINER_H

#include <vector>

#include "ndim/pointer.h"

namespace ndim
{

template <typename _ElementType>
class ContainerVar;

template <typename _ElementType>
class ContainerBase
{
public:
    using ElementType = _ElementType;

protected:
    std::vector<ElementType> m_vector;

public:
    ContainerBase()
    {
    }
    ~ContainerBase() = default;
    ContainerBase(const ContainerBase<ElementType> &other) = delete;
    ContainerBase(ContainerBase<ElementType> &&other) = default;

    /**
     * @brief The capacity of the owned buffer.
     * @return
     */
    size_t capacity() const
    {
        return m_vector.capacity();
    }

    /**
     * @brief Checks if the container owns a buffer.
     * @return
     */
    bool ownsData() const
    {
        return !m_vector.empty();
    }

    /**
     * @brief Swaps buffer-ownership between two containers. Data references stay same.
     * @param other
     */
    void swapOwnership(ContainerBase<_ElementType> &other)
    {
        std::swap(this->m_vector, other.m_vector);
    }
};

template <typename _ElementType, size_t _Dimensionality = 0>
class Container : public ContainerBase<_ElementType>
{
public:
    using ElementType = _ElementType;
    static const size_t Dimensionality = _Dimensionality;

private:
    ElementType *m_mutableData;
    const ElementType *m_constData;
    ndim::layout<Dimensionality> m_layout;

    friend class ContainerVar<ElementType>;

public:
    Container()
        : m_mutableData(nullptr)
        , m_constData(nullptr)
    {
    }

    /**
     * @brief If you need a copy you did something wrong.
     * @param other
     */
    Container(const Container<ElementType, Dimensionality> &other) = delete;

    /**
     * @brief Move constructor.
     * @param other
     */
    Container(Container<ElementType, Dimensionality> &&other)
        : ContainerBase<ElementType>(std::move(other))
        , m_mutableData(other.m_mutableData)
        , m_constData(other.m_constData)
        , m_layout(other.m_layout)
    {
        other.m_mutableData = nullptr;
        other.m_constData = nullptr;
    }

    /**
     * @brief Creates a mutable data reference without ownership.
     * @param data
     */
    Container(ndim::pointer<ElementType, Dimensionality> data)
        : m_mutableData(data.data)
        , m_constData(data.data)
        , m_layout(data.getLayout())
    {
    }

    /**
     * @brief Creates a const data reference without ownership.
     * @param data
     */
    Container(ndim::pointer<const ElementType, Dimensionality> data)
        : m_mutableData(nullptr)
        , m_constData(data.data)
        , m_layout(data.getLayout())
    {
    }

    /**
     * @brief If you need a copy you did something wrong.
     * @param other
     * @return
     */
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
    const ndim::layout<Dimensionality> &layout() const
    {
        return m_layout;
    }
};

template <typename _ElementType>
class ContainerVar : public ContainerBase<_ElementType>
{
public:
    using ElementType = _ElementType;

private:
    ElementType *m_mutableData;
    const ElementType *m_constData;
    ndim::LayoutVar m_layout;

public:
    ContainerVar()
        : m_mutableData(nullptr)
        , m_constData(nullptr)
    {
    }

    /**
     * @brief If you need a copy you did something wrong.
     * @param other
     */
    ContainerVar(const ContainerVar<ElementType> &other) = delete;

    /**
     * @brief Move constructor.
     * @param other
     */
    ContainerVar(ContainerVar<ElementType> &&other)
        : ContainerBase<ElementType>(std::move(other))
        , m_mutableData(other.m_mutableData)
        , m_constData(other.m_constData)
        , m_layout(other.m_layout)
    {
        other.m_mutableData = nullptr;
        other.m_constData = nullptr;
    }

    /**
     * @brief Creates a mutable data reference without ownership.
     * @param data
     */
    ContainerVar(PointerVar<ElementType> data)
        : m_mutableData(data.data)
        , m_constData(data.data)
        , m_layout(std::move(data))
    {
    }

    /**
     * @brief Creates a const data reference without ownership.
     * @param data
     */
    ContainerVar(PointerVar<const ElementType> data)
        : m_mutableData(nullptr)
        , m_constData(data.data)
        , m_layout(std::move(data))
    {
    }

    /**
     * @brief If you need a copy you did something wrong.
     * @param other
     * @return
     */
    ContainerVar<ElementType> &operator=(const ContainerVar<ElementType> &other) = delete;

    ContainerVar<ElementType> &operator=(ContainerVar<ElementType> &&other)
    {
        this->m_vector = std::move(other.m_vector);
        this->m_mutableData = other.m_mutableData;
        this->m_constData = other.m_constData;
        this->m_layout = std::move(other.m_layout);
        other.m_mutableData = nullptr;
        other.m_constData = nullptr;

        return *this;
    }

    /**
     * @brief Move from fixed dimensional container.
     */
    template <size_t Dimensionality>
    ContainerVar(Container<ElementType, Dimensionality> &&other)
        : ContainerBase<ElementType>(std::move(other))
        , m_mutableData(other.m_mutableData)
        , m_constData(other.m_constData)
        , m_layout(other.m_layout)
    {
        other.m_mutableData = nullptr;
        other.m_constData = nullptr;
    }

    template <size_t Dimensionality>
    Container<ElementType, Dimensionality> fixDimensionality()
    {
        Container<ElementType, Dimensionality> result;
        result.m_layout = this->m_layout.template fixDimensionality<Dimensionality>();
        result.m_vector = std::move(this->m_vector);
        result.m_mutableData = this->m_mutableData;
        result.m_constData = this->m_constData;

        this->m_mutableData = nullptr;
        this->m_constData = nullptr;

        return result;
    }

    void changePointer(ndim::PointerVar<ElementType> data)
    {
        m_constData = m_mutableData = data.data;
        m_layout = std::move(data);
    }
    void changePointer(ndim::PointerVar<const ElementType> data)
    {
        m_mutableData = nullptr;
        m_constData = data.data;
        m_layout = std::move(data);
    }

    void resize(ShapeVar shape)
    {
        this->m_vector.resize(ndim::totalCount(shape));
        m_constData = m_mutableData = this->m_vector.data();
        m_layout = LayoutVar(shape, hlp::byte_offset_t::inArray<ElementType>());
    }

    bool hasData() const
    {
        return m_constData;
    }
    bool isMutable() const
    {
        return m_mutableData;
    }
    ndim::PointerVar<ElementType> mutableData() const
    {
        return ndim::make_pointer(m_mutableData, m_layout);
    }
    ndim::PointerVar<const ElementType> constData() const
    {
        return ndim::make_pointer(m_constData, m_layout);
    }
    const LayoutVar &layout() const
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
