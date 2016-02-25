#ifndef FC_FILTER_SUBRANGE_H
#define FC_FILTER_SUBRANGE_H

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "ndim/range.h"

namespace fc
{
namespace filter
{

template <size_t _Dimensionality>
class SubrangeControl
{
	static const size_t Dimensionality = _Dimensionality;

public:
	virtual ndim::range<Dimensionality> range() const = 0;
	virtual void setRange(ndim::range<Dimensionality> range) = 0;
};

template <typename _ElementType, size_t _Dimensionality>
class SubrangeHandler : public DataFilterHandlerBase<const DataFilter<_ElementType, _Dimensionality>>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;
	using BaseType = DataFilterHandlerBase<const DataFilter<ElementType, Dimensionality>>;

	ndim::range<Dimensionality> range;

public:
	ndim::sizes<Dimensionality> prepare(PreparationProgress &progress) const
	{
		this->preparePredecessors(progress);
		return this->range.sizes;
	}

	Container<ElementType, Dimensionality> getData(ValidationProgress &progress, Container<ElementType, Dimensionality> *recycle) const
	{
		Container<ElementType, Dimensionality> result = std::get<0>(this->getPredecessorsData(progress, recycle));
		if (result.isMutable())
			result.changePointer(result.mutableData().selectedRange(range));
		else
			result.changePointer(result.constData().selectedRange(range));
		return result;
	}
};

template <typename _ElementType, size_t _Dimensionality>
class Subrange : public fc::HandlerDataFilterBase<_ElementType, _Dimensionality, SubrangeHandler<_ElementType, _Dimensionality>>,
				 public virtual SubrangeControl<_Dimensionality>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

public:
	ndim::range<Dimensionality> range() const override
	{
		return this->m_handler.unguarded().range;
	}
	void setRange(ndim::range<Dimensionality> range) override
	{
		if (this->m_handler.unguarded().range == range)
			return;
		this->invalidate();
		this->m_handler.lock()->range = range;
	}
};

template <typename _Predecessor>
std::shared_ptr<Subrange<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value>> makeSubrange(
	std::shared_ptr<_Predecessor> predecessor,
	ndim::range<DimensionalityOf_t<_Predecessor>::value> range = ndim::range<DimensionalityOf_t<_Predecessor>::value>())
{
	auto filter = std::make_shared<Subrange<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value>>();
	filter->setRange(range);
	filter->setPredecessor(predecessor);
	return filter;
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_SUBRANGE_H
