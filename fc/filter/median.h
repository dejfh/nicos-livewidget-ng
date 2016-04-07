#ifndef FC_FILTER_MEDIAN_H
#define FC_FILTER_MEDIAN_H

#include <numeric>

#include "fc/datafilterbase.h"
#include "fc/gethelper.h"

#include "ndim/iterator.h"
#include "ndim/strideiterator.h"
#include "ndim/mean.h"

#include "helper/helper.h"
#include "helper/array.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, size_t _PredecessorDimensionality, size_t _ResultDimensionality>
class MedianHandler : public DataFilterHandlerBase<const DataFilter<_ElementType, _PredecessorDimensionality>>
{
public:
	using ElementType = _ElementType;
	using ResultElementType = _ElementType;
	static const size_t PredecessorDimensionality = _PredecessorDimensionality;
	static const size_t ResultDimensionality = _ResultDimensionality;
	static const size_t MedianDimensionality = PredecessorDimensionality - ResultDimensionality;

	std::array<size_t, ResultDimensionality> resultDimensions;
	float dropAmountLow, dropAmountHigh;
	QString description;

private:
	static std::pair<size_t, size_t> getStartAndEnd(size_t medianCount, float dropAmountLow, float dropAmountHigh)
	{
		// TODO: medianCount == 0?

		size_t dropCountLow = dropAmountLow * medianCount + .5f;
		size_t dropCountHigh = dropAmountHigh * medianCount + .5f;

		if (dropCountLow + dropCountHigh >= medianCount) {
			size_t index = dropAmountLow / (dropAmountLow + dropAmountHigh) * medianCount;
			index = std::min(index, medianCount - 1);
			dropCountLow = index;
			dropCountHigh = medianCount - index - 1;
		}
		return std::make_pair(dropCountLow, medianCount - dropCountHigh);
	}

public:
	ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const
	{
		auto sizes = this->preparePredecessors(progress);
		ndim::Sizes<PredecessorDimensionality> &predecessorSizes = std::get<0>(sizes);

		auto medianDimensions = hlp::array::invertSelection<PredecessorDimensionality>(resultDimensions);
		auto medianSizes = hlp::array::select(predecessorSizes, medianDimensions);

		auto medianBounds = getStartAndEnd(medianSizes.size(), dropAmountLow, dropAmountHigh);
		size_t medianDuration = ndim::inPlaceMeanDuration(predecessorSizes, resultDimensions, medianBounds.first, medianBounds.second);

		progress.addStep(medianDuration, description);

		return hlp::array::select(predecessorSizes, resultDimensions);
	}

	ndim::Container<ResultElementType, ResultDimensionality> getData(
		ValidationProgress &progress, ndim::Container<ResultElementType, ResultDimensionality> *recycle) const
	{
		auto container = this->getPredecessorsData(progress, (void *)nullptr);

		ndim::Container<ElementType, PredecessorDimensionality> &input = std::get<0>(container);
		ndim::pointer<const ElementType, PredecessorDimensionality> constData = input.constData();
		ndim::pointer<ElementType, PredecessorDimensionality> mutableData;

		ndim::Container<ElementType, PredecessorDimensionality> temp;
		ndim::Sizes<MedianDimensionality> medianDimensions = hlp::array::invertSelection<PredecessorDimensionality>(resultDimensions);

		if (input.isMutable() && constData.selectDimensions(medianDimensions).isContiguous())
			mutableData = input.mutableData();
		else {
			// reorder median dimensions to front to make them contiguous.
			auto reorderedDimensions = hlp::array::cat(medianDimensions, resultDimensions);
			temp.resize(constData.sizes);
			// get ndim::pointer in original dimensional order
			mutableData = temp.mutableData().selectDimensions(hlp::array::invertReorder(reorderedDimensions));
			ndim::copy(constData, mutableData);
		}

		ndim::Container<ElementType, ResultDimensionality> result =
			ndim::makeMutableContainer(constData.sizes.selectDimensions(resultDimensions), recycle);

		ndim::sizes<MedianDimensionality> medianSizes = constData.sizes.selectDimensions(medianDimensions);

		size_t medianCount = medianSizes.size();

		std::pair<size_t, size_t> bounds = getStartAndEnd(medianCount, dropAmountLow, dropAmountHigh);
		ndim::inPlaceMean(mutableData, result.mutableData(), resultDimensions, bounds.first, bounds.second, progress);

		progress.advanceStep();

		return result;
	}
};

template <typename _ElementType, size_t _PredecessorDimensionality, size_t _ResultDimensionality>
class Median : public HandlerDataFilterBase<_ElementType, _ResultDimensionality,
				   MedianHandler<_ElementType, _PredecessorDimensionality, _ResultDimensionality>>
{
public:
	static const size_t ResultDimensionality = _ResultDimensionality;

	Median(std::array<size_t, ResultDimensionality> resultDimensions, float dropAmountLow, float dropAmountHigh, const QString &description)
	{
		auto guard = this->m_handler.lock();
		guard->resultDimensions = resultDimensions;
		guard->dropAmountLow = dropAmountLow;
		guard->dropAmountHigh = dropAmountHigh;
		guard->description = description;
	}
};

template <typename ElementType, size_t PredecessorDimensionality, size_t ResultDimensionality>
std::shared_ptr<Median<ElementType, PredecessorDimensionality, ResultDimensionality>> _makeMedian(QString description,
	std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>> predecessor, std::array<size_t, ResultDimensionality> resultDimensions,
	float start = .2, float end = .8)
{
	auto filter =
		std::make_shared<Median<ElementType, PredecessorDimensionality, ResultDimensionality>>(resultDimensions, start, 1.f - end, description);
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

template <typename ElementType, size_t PredecessorDimensionality>
std::shared_ptr<Median<ElementType, PredecessorDimensionality, PredecessorDimensionality - 1>> _makeMedian(QString description,
	std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>> predecessor, size_t medianDimension, float start = .2, float end = .8)
{
	std::array<size_t, 1> medianDimensions = {medianDimension};
	return _makeMedian(
		std::move(description), std::move(predecessor), hlp::array::invertSelection<PredecessorDimensionality>(medianDimensions), start, 1.f - end);
}

template <typename PredecessorType, size_t ResultDimensionality>
auto makeMedian(QString description, std::shared_ptr<PredecessorType> predecessor, std::array<size_t, ResultDimensionality> resultDimensions,
	float start = .2, float end = .8) -> decltype(_makeMedian(std::move(description), std::move(predecessor), resultDimensions, start, end))
{
	return _makeMedian(std::move(description), std::move(predecessor), resultDimensions, start, end);
}

template <typename PredecessorType>
auto makeMedian(QString description, std::shared_ptr<PredecessorType> predecessor, size_t medianDimension, float start = .2, float end = .8)
	-> decltype(_makeMedian(std::move(description), asDataFilter(std::move(predecessor)), medianDimension, start, end))
{
	return _makeMedian(std::move(description), asDataFilter(std::move(predecessor)), medianDimension, start, end);
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_MEDIAN_H
