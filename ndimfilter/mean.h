#ifndef NDIMFILTER_MEAN_H
#define NDIMFILTER_MEAN_H

#include <numeric>

#include "filter/gethelper.h"
#include "ndimfilter/filter.h"
#include "ndim/iterator.h"
#include "ndim/buffer.h"
#include "ndim/strideiterator.h"
#include "ndim/mean.h"

#include "helper/helper.h"

namespace filter
{

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _ElementType, size_t _PredecessorDimensionality, size_t _ResultDimensionality>
class mean : public SinglePredecessorFilterBase<DataFilter<_ElementType, _PredecessorDimensionality>>,
			 public NoConstDataFilter<_ElementType, _ResultDimensionality>
{
public:
	using ElementType = _ElementType;
	static const size_t PredecessorDimensionality = _PredecessorDimensionality;
	static const size_t ResultDimensionality = _ResultDimensionality;
	static const size_t MedianDimensionality = PredecessorDimensionality - ResultDimensionality;

private:
	std::array<size_t, ResultDimensionality> m_resultDimensions;
	float m_dropAmountLow, m_dropAmountHigh;
	const QString m_description;

	mutable ndim::sizes<PredecessorDimensionality> m_predecessorSizes;

public:
	mean(std::array<size_t, ResultDimensionality> resultDimensions, float dropAmountLow, float dropAmountHigh, const QString &description)
		: m_resultDimensions(resultDimensions)
		, m_dropAmountLow(dropAmountLow)
		, m_dropAmountHigh(dropAmountHigh)
		, m_description(description)
	{
	}

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

	// DataFilter interface
public:
	virtual ndim::sizes<ResultDimensionality> prepare(
		PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, ResultDimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto resultDimensions = m_resultDimensions;
		float dropAmountLow = m_dropAmountLow;
		float dropAmountHigh = m_dropAmountHigh;
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		m_predecessorSizes = predecessor->prepare(progress);

		auto medianDimensions = hlp::array::invertSelection<PredecessorDimensionality>(resultDimensions);
		auto medianSizes = m_predecessorSizes.selectDimensions(medianDimensions);

		auto medianBounds = getStartAndEnd(medianSizes.size(), dropAmountLow, dropAmountHigh);
		size_t medianDuration = ndim::inPlaceMeanDuration(m_predecessorSizes, resultDimensions, medianBounds.first, medianBounds.second);
		progress.addStep(medianDuration, m_description);

		return m_predecessorSizes.selectDimensions(resultDimensions);
	}
	virtual void getData(ValidationProgress &progress, Container<ElementType, ResultDimensionality> &result,
		OverloadDummy<DataFilter<ElementType, ResultDimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto resultDimensions = m_resultDimensions;
		float dropAmountLow = m_dropAmountLow;
		float dropAmountHigh = m_dropAmountHigh;
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		auto medianDimensions = hlp::array::invertSelection<PredecessorDimensionality>(resultDimensions);

		// reorder median dimensions to front to make them contiguous.
		auto reorderedDimensions = hlp::array::cat(medianDimensions, resultDimensions);
		auto medianSizes = m_predecessorSizes.selectDimensions(medianDimensions);
		ndim::Buffer<ElementType, PredecessorDimensionality> buffer;
		buffer.resize(m_predecessorSizes);
		// get ndim::pointer in original dimensional order
		ndim::pointer<ElementType, PredecessorDimensionality> pointer = buffer.pointer().selectDimensions(hlp::array::invertReorder(reorderedDimensions));

		filter::getData(progress, predecessor, pointer);

		result.resize(m_predecessorSizes.selectDimensions(resultDimensions));

		std::pair<size_t, size_t> bounds = getStartAndEnd(medianSizes.size(), dropAmountLow, dropAmountHigh);
		ndim::inPlaceMean(pointer, result.pointer(), resultDimensions, bounds.first, bounds.second, progress);

		progress.advanceStep();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename ElementType, size_t PredecessorDimensionality, size_t ResultDimensionality>
std::shared_ptr<mean<ElementType, PredecessorDimensionality, ResultDimensionality>> _makeMean(QString description,
	std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>> predecessor, std::array<size_t, ResultDimensionality> resultDimensions,
	float start = .2, float end = .8)
{
	auto filter =
		std::make_shared<mean<ElementType, PredecessorDimensionality, ResultDimensionality>>(resultDimensions, start, 1.f - end, description);
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

template <typename ElementType, size_t PredecessorDimensionality>
std::shared_ptr<mean<ElementType, PredecessorDimensionality, PredecessorDimensionality - 1>> _makeMean(QString description,
	std::shared_ptr<const DataFilter<ElementType, PredecessorDimensionality>> predecessor, size_t medianDimension, float start = .2, float end = .8)
{
	std::array<size_t, 1> medianDimensions = {medianDimension};
	return _makeMean(
		std::move(description), std::move(predecessor), hlp::array::invertSelection<PredecessorDimensionality>(medianDimensions), start, 1.f - end);
}

template <typename PredecessorType, size_t ResultDimensionality>
auto makeMean(QString description, std::shared_ptr<PredecessorType> predecessor, std::array<size_t, ResultDimensionality> resultDimensions,
	float start = .2, float end = .8) -> decltype(_makeMean(std::move(description), std::move(predecessor), resultDimensions, start, end))
{
	return _makeMean(std::move(description), std::move(predecessor), resultDimensions, start, end);
}

template <typename PredecessorType>
auto makeMean(QString description, std::shared_ptr<PredecessorType> predecessor, size_t medianDimension, float start = .2, float end = .8)
	-> decltype(_makeMean(std::move(description), std::move(predecessor), medianDimension, start, end))
{
	return _makeMean(std::move(description), std::move(predecessor), medianDimension, start, end);
}

} // namespace ndimfilter

#endif // NDIMFILTER_MEAN_H
