#ifndef DATA_FILTERBUFFER_H
#define DATA_FILTERBUFFER_H

#include <exception>
#include <atomic>

#include "filter/filter.h"
#include "filter/filterbase.h"
#include "filter/typetraits.h"

#include "helper/helper.h"

#include "ndim/algorithm_omp.h"

namespace filter
{

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _ElementType, size_t _Dimensionality = 0>
class Buffer : public SinglePredecessorFilterBase<DataFilter<_ElementType, _Dimensionality>>,
			   public virtual DataFilter<_ElementType, _Dimensionality>,
			   public virtual Validatable
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

protected:
	mutable Container<const ElementType, Dimensionality> m_data;
	mutable std::atomic<bool> m_isValid;

public:
	Buffer()
		: m_isValid(false)
	{
	}

	ndim::pointer<const ElementType, Dimensionality> data() const
	{
		if (!m_isValid)
			throw std::logic_error("Buffer is not valid.");
		return m_data.pointer();
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
	virtual void prepare(PreparationProgress &progress) const override
	{
		if (m_isValid || progress.containsValidatable(this)) {
			return;
		}
		auto predecessor = this->predecessor();
		progress.throwIfCancelled();
		ndim::sizes<Dimensionality> sizes = hlp::notNull(predecessor)->prepareConst(progress);
		m_data.reset(ndim::pointer<const ElementType, Dimensionality>(nullptr, sizes));
		progress.appendValidatable(std::shared_ptr<const Validatable>(this->shared_from_this(), this));
	}
	virtual void validate(ValidationProgress &progress) const override
	{
		if (m_isValid)
			return;
		auto predecessor = this->predecessor();
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		predecessor->getConstData(progress, m_data);
		m_isValid = true;
	}
	virtual bool isValid() const override
	{
		return m_isValid;
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		this->prepare(progress);
		return m_data.pointer().sizes;
	}
	virtual ndim::sizes<Dimensionality> prepareConst(
		PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		this->prepare(progress);
		return m_data.pointer().sizes;
	}
	virtual void getData(ValidationProgress &progress, Container<ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		this->validate(progress);
		ndim::pointer<const ElementType, Dimensionality> pointer = m_data.pointer();
		result.resize(pointer.sizes);
#pragma omp parallel
		{
			ndim::copy_omp(pointer, result.pointer());
		}
	}
	virtual void getConstData(ValidationProgress &progress, Container<const ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		this->validate(progress);
		result.reset(m_data.pointer());
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename _FilterType>
std::shared_ptr<Buffer<ElementTypeOf_t<_FilterType>, DimensionalityOf_t<_FilterType>::value>> makeBuffer(std::shared_ptr<_FilterType> predecessor)
{
	auto filter = std::make_shared<Buffer<ElementTypeOf_t<_FilterType>, DimensionalityOf_t<_FilterType>::value>>();
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace filter

#endif // DATA_FILTERBUFFER_H
