#ifndef FILTER_FS_BUFFER_H
#define FILTER_FS_BUFFER_H

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

	mutable Container<ElementType, Dimensionality> m_data;
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
		this->invalidate();
		this->unregisterSuccessor(m_predecessor.unguarded());
		this->registerSuccessor(predecessor);
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
		this->invalidate();
		m_isValid = false;
	}

	// Validatable interface
public:
	virtual void prepareValidation(PreparationProgress &progress) const override
	{
		if (m_isValid || progress.containsValidatable(this)) {
			return;
		}
		auto predecessor = this->predecessor();
		hlp::notNull(predecessor);
		progress.throwIfCancelled();
		ndim::sizes<Dimensionality> sizes = hlp::notNull(predecessor)->prepare(progress);
		m_data.setMutablePointer(ndim::pointer<ElementType, Dimensionality>(nullptr, sizes));
		progress.appendValidatable(std::shared_ptr<const Validatable>(this->shared_from_this(), this));
	}
	virtual void validate(ValidationProgress &progress) const override
	{
		if (m_isValid)
			return;
		auto predecessor = this->predecessor();
		progress.throwIfCancelled();

		m_data = predecessor->getData(progress, &m_data);
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
		this->prepare(progress);
		return m_data.layout().sizes;
	}
	virtual Container<ElementType, Dimensionality> getData(
		ValidationProgress &progress, Container<ElementType, Dimensionality> *recycle) const override
	{
		hlp::unused(recycle);
		this->validate(progress);
		return fc::makeConstRefContainer(m_data.constData());
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename PredecessorType>
std::shared_ptr<Buffer<ElementTypeOf_t<PredecessorType>, DimensionalityOf_t<PredecessorType>::value>> makeBuffer(std::shared_ptr<PredecessorType> predecessor)
{
	auto filter = std::make_shared<Buffer<ElementTypeOf_t<PredecessorType>, DimensionalityOf_t<PredecessorType>::value>>();
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace filter
} // namespace fc

#endif // FILTER_FS_BUFFER_H
