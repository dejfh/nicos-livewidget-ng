#ifndef FC_FILTER_BUFFER_H
#define FC_FILTER_BUFFER_H

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "helper/helper.h"
#include "helper/threadsafe.h" // hlp::Threadsafe

#include "ndim/algorithm_omp.h"

namespace fc
{
namespace filter
{

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _ElementType, size_t _Dimensionality = 0>
class Buffer : public FilterBase, public virtual DataFilter<_ElementType, _Dimensionality>, public virtual Validatable
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

protected:
	hlp::Threadsafe<std::shared_ptr<const DataFilter<ElementType, Dimensionality>>> m_predecessor;

	mutable ndim::Container<ElementType, Dimensionality> m_data;
	mutable std::atomic<bool> m_isValid;

public:
	Buffer()
		: m_isValid(false)
	{
	}

	std::shared_ptr<const DataFilter<ElementType, Dimensionality>> predecessor() const
	{
		return m_predecessor.get();
	}
	void setPredecessor(std::shared_ptr<const DataFilter<ElementType, Dimensionality>> predecessor)
	{
		if (m_predecessor.unguarded() == predecessor)
			return;
		m_isValid = false;
		this->invalidate();
		this->unregisterAsSuccessor(m_predecessor.unguarded());
		this->registerAsSuccessor(predecessor);
		m_predecessor = std::move(predecessor);
	}

	ndim::pointer<const ElementType, Dimensionality> data() const
	{
		if (!m_isValid)
			throw std::logic_error("Buffer is not valid.");
		return m_data.constData();
	}

	// Invalidatable interface
public:
	virtual void predecessorInvalidated(const Predecessor *) override
	{
		// Order is important: isValid has to be false during callbacks invoked by invalidate.
		m_isValid = false;
		this->invalidate();
	}

	// Validatable interface
public:
	virtual void prepareValidation(PreparationProgress &progress) const override
	{
		if (m_isValid || progress.containsValidatable(this)) {
			return;
		}
		auto predecessor = this->predecessor();
		hlp::throwIfNull(predecessor);
		progress.throwIfCancelled();
		ndim::sizes<Dimensionality> sizes = hlp::throwIfNull(predecessor)->prepare(progress);
		m_data = ndim::pointer<ElementType, Dimensionality>(nullptr, sizes);
		progress.appendValidatable(std::shared_ptr<const Validatable>(this->shared_from_this(), this));
	}
	virtual void validate(ValidationProgress &progress) const override
	{
		progress.holdRef(this->shared_from_this());
		if (m_isValid)
			return;
		auto predecessor = this->predecessor();
		progress.throwIfCancelled();

		auto data = predecessor->getData(progress, &m_data);
		if (data.ownsData())
			m_data = std::move(data);
		else // if (!data.ownsData())
		{
			// TODO Would be better to keep reference, instead of copying, but the owning filter could be destructed during subsequent validations, if
			// predecessors are changed.
			auto ptr = data.constData();
			m_data.resize(ptr.sizes);
#pragma omp parallel
			{
				ndim::copy_omp(ptr, m_data.mutableData());
			}
		}
		m_isValid = true;
	}
	virtual bool isValid() const override
	{
		return m_isValid;
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress) const override
	{
		this->prepareValidation(progress);
		return m_data.layout().sizes;
	}
	virtual ndim::Container<ElementType, Dimensionality> getData(
		ValidationProgress &progress, ndim::Container<ElementType, Dimensionality> *recycle) const override
	{
		hlp::unused(recycle);
		this->validate(progress);
		return m_data.constData();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename PredecessorType>
std::shared_ptr<Buffer<ElementTypeOf_t<PredecessorType>, DimensionalityOf_t<PredecessorType>::value>> makeBuffer(
	std::shared_ptr<PredecessorType> predecessor)
{
	auto filter = std::make_shared<Buffer<ElementTypeOf_t<PredecessorType>, DimensionalityOf_t<PredecessorType>::value>>();
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_BUFFER_H
