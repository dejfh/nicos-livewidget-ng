#ifndef NDIMFILTER_RANGE_H
#define NDIMFILTER_RANGE_H

#include "ndim/range.h"

#include "filter/filter.h"
#include "filter/filterbase.h"
#include "filter/gethelper.h"
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

template <size_t _Dimensionality>
class RangeControl
{
	static const size_t Dimensionality = _Dimensionality;

public:
	virtual ndim::range<Dimensionality> selectedRange() const = 0;
	virtual void setRange(ndim::range<Dimensionality> range) = 0;
};

template <typename _ElementType, size_t _Dimensionality>
class range : public filter::SinglePredecessorFilterBase<DataFilter<_ElementType, _Dimensionality>>,
			  public virtual DataFilter<_ElementType, _Dimensionality>,
			  public virtual RangeControl<_Dimensionality>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	ndim::range<Dimensionality> m_range;

public:
	range(ndim::range<Dimensionality> range = ndim::range<Dimensionality>())
		: m_range(range)
	{
	}

	ndim::range<Dimensionality> selectedRange() const override
	{
		return m_range;
	}
	void setRange(ndim::range<Dimensionality> range) override
	{
		this->setAndInvalidate(m_range, range);
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto range = m_range;
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		predecessor->prepare(progress);
		return range.sizes;
	}
	virtual ndim::sizes<Dimensionality> prepareConst(
		PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto range = m_range;
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		predecessor->prepareConst(progress);
		return range.sizes;
	}
	virtual void getData(ValidationProgress &progress, Container<ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto range = m_range;
		progress.throwIfCancelled();

		predecessor->getData(progress, result);
		result.m_pointer.selectRange(range);
	}
	virtual void getConstData(ValidationProgress &progress, Container<const ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto range = m_range;
		progress.throwIfCancelled();

		predecessor->getConstData(progress, result);
		result.m_pointer.selectRange(range);
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename _Predecessor>
std::shared_ptr<filter::range<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value>> makeRange(
	std::shared_ptr<_Predecessor> predecessor,
	ndim::range<DimensionalityOf_t<_Predecessor>::value> range = ndim::range<DimensionalityOf_t<_Predecessor>::value>())
{
	auto filter = std::make_shared<filter::range<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value>>(range);
	filter->setPredecessor(predecessor);
	return filter;
}

} // namespace ndimfilter

#endif // NDIMFILTER_RANGE_H
