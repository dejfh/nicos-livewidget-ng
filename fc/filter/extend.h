#ifndef FC_FILTER_EXTEND_H
#define FC_FILTER_EXTEND_H

#include <array>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _PredecessorDimensionality, size_t _ResultDimensionality>
class ExtendHandler : public DataFilterHandlerBase<const DataFilter<_ElementType, _PredecessorDimensionality>>
{
public:
	using ElementType = _ElementType;
	static const size_t PredecessorDimensionality = _PredecessorDimensionality;

	using ResultElementType = ElementType;
	static const size_t ResultDimensionality = _ResultDimensionality;
	static const size_t ExtendDimensionality = ResultDimensionality - PredecessorDimensionality;

	static_assert(PredecessorDimensionality <= ResultDimensionality, "Can not extend to lower dimensionality.");

	ndim::Indices<ExtendDimensionality> extendDimensions;
	ndim::Sizes<ExtendDimensionality> extendSizes;

	ExtendHandler(ndim::Indices<ExtendDimensionality> extendDimensions, ndim::Sizes<ExtendDimensionality> extendSizes)
		: extendDimensions(extendDimensions)
		, extendSizes(extendSizes)
	{
	}

	ndim::Sizes<ResultDimensionality> prepare(PreparationProgress &progress) const
	{
		ndim::Sizes<PredecessorDimensionality> sizes = std::get<0>(this->preparePredecessors(progress));

		return hlp::array::insert(sizes, extendSizes, extendDimensions);
	}

	Container<ElementType, ResultDimensionality> getData(ValidationProgress &progress, Container<ElementType, ResultDimensionality> *recycle) const
	{
		fc::Container<ElementType, PredecessorDimensionality> temp;
		if (recycle)
			temp.setOwnership(std::move(*recycle));
		Container<ElementType, PredecessorDimensionality> input = std::get<0>(this->getPredecessorsData(progress, &temp));
		if (recycle)
			recycle->setOwnership(std::move(temp));

		Container<ElementType, ResultDimensionality> result(input.takeOwnership());
		result.setConstPointer(input.constData().addDimensions(extendDimensions, extendSizes));

		return result;
	}
};

template <typename _ElementType, size_t _PredecessorDimensionality, size_t _ResultDimensionality>
class Extend : public HandlerDataFilterBase<_ElementType, _ResultDimensionality,
				   ExtendHandler<_ElementType, _PredecessorDimensionality, _ResultDimensionality>>
{
public:
	using HandlerType = ExtendHandler<_ElementType, _PredecessorDimensionality, _ResultDimensionality>;

	static const size_t PredecessorDimensionality = _PredecessorDimensionality;
	static const size_t ResultDimensionality = _ResultDimensionality;
	static const size_t ExtendDimensionality = ResultDimensionality - PredecessorDimensionality;

	Extend(ndim::Indices<ExtendDimensionality> extendDimensions, ndim::Sizes<ExtendDimensionality> extendSizes)
		: HandlerDataFilterBase<_ElementType, _ResultDimensionality, HandlerType>(HandlerType(extendDimensions, extendSizes))
	{
	}

	ndim::Indices<ExtendDimensionality> extendDimensions() const
	{
		return this->m_handler.unguarded().extendDimensions;
	}
	void setExtendDimensions(ndim::Indices<ExtendDimensionality> extendDimensions)
	{
		if (this->m_handler.unguarded().extendDimensions == extendDimensions)
			return;
		this->invalidate();
		this->m_handler.lock()->extendDimensions = extendDimensions;
	}

	ndim::Sizes<ExtendDimensionality> extendSizes() const
	{
		return this->m_handler.unguarded().extendSizes;
	}
	void setExtendSizes(ndim::Sizes<ExtendDimensionality> extendSizes)
	{
		if (this->m_handler.unguarded().extendSizes == extendSizes)
			return;
		this->invalidate();
		this->m_handler.lock()->extendSizes = extendSizes;
	}
};

template <typename PredecessorType, size_t ExtendDimensionality>
std::shared_ptr<Extend<ElementTypeOf_t<PredecessorType>, DimensionalityOf_t<PredecessorType>::value,
	DimensionalityOf_t<PredecessorType>::value + ExtendDimensionality>>
makeExtend(
	std::shared_ptr<PredecessorType> predecessor, ndim::Indices<ExtendDimensionality> extendDimensions, ndim::Sizes<ExtendDimensionality> extendSizes)
{
	auto filter = std::make_shared<Extend<ElementTypeOf_t<PredecessorType>, DimensionalityOf_t<PredecessorType>::value,
		DimensionalityOf_t<PredecessorType>::value + ExtendDimensionality>>(extendDimensions, extendSizes);
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_EXTEND_H
