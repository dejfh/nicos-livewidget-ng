#ifndef FC_FILTER_ACCUMULATE_H
#define FC_FILTER_ACCUMULATE_H

#include <array>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "ndim/algorithm_omp.h"

namespace fc
{
namespace filter
{

template <typename _PredecessorElementType, size_t _PredecessorDimensionality, typename _ResultElementType, size_t _ResultDimensionality,
	typename _OperationType>
class AccumulateHandler : public DataFilterHandlerBase<const DataFilter<_PredecessorElementType, _PredecessorDimensionality>>
{
	using PredecessorElementType = _PredecessorElementType;
	static const size_t PredecessorDimensionality = _PredecessorDimensionality;

	using ResultElementType = _ResultElementType;
	static const size_t ResultDimensionality = _ResultDimensionality;

	static const size_t DimensionalityDifference = PredecessorDimensionality - ResultDimensionality;

	using OperationType = _OperationType;

public:
	OperationType operation;
	ndim::Indices<ResultDimensionality> selectedDimensions;
	QString description;

	AccumulateHandler(OperationType operation, ndim::Indices<ResultDimensionality> selectedDimensions, const QString &description)
		: operation(std::move(operation))
		, selectedDimensions(selectedDimensions)
		, description(std::move(description))
	{
	}

	ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const
	{
		auto sizes = this->preparePredecessors(progress);

		auto &sizes0 = std::get<0>(sizes);

		progress.addStep(ndim::totalCount(sizes0), description);

		return hlp::array::select(sizes0, selectedDimensions);
	}
	Container<ResultElementType, ResultDimensionality> getData(
		ValidationProgress &progress, Container<ResultElementType, ResultDimensionality> *recycle) const
	{
		auto inputs = this->getPredecessorsData(progress, recycle);
		Container<PredecessorElementType, PredecessorDimensionality> &input = std::get<0>(inputs);
		auto inputPointer = input.constData();
		auto selectedPointer = inputPointer.selectDimensions(selectedDimensions);

		auto accumulateDimensions = hlp::array::invertSelection<PredecessorDimensionality>(selectedDimensions);
		ndim::layout<DimensionalityDifference> accumulateLayout = inputPointer.getLayout().selectDimensions(accumulateDimensions);

		Container<ResultElementType, ResultDimensionality> result = makeMutableContainer(selectedPointer.sizes, recycle);

		auto resultPointer = result.mutableData();

#pragma omp parallel
		{
			// Initiate result to default.
			ndim::operate_omp([](ResultElementType &value) { value = ResultElementType(); }, resultPointer);
#pragma omp barrier
			auto inIterator = ndim::makeIterator(selectedPointer);
			auto outIterator = ndim::makeIterator(resultPointer);
			for (size_t count = selectedPointer.size(); count > 0; --count, ++inIterator, ++outIterator) {
				ndim::pointer<const PredecessorElementType, DimensionalityDifference> accPointer(&*inIterator, accumulateLayout);
				ResultElementType acc = ndim::accumulate_omp(operation, ResultElementType(), accPointer);
#pragma omp critical
				{
					*outIterator = operation(*outIterator, acc);
				}
			}
		}

		progress.advanceProgress(accumulateLayout.size());
		progress.advanceStep();

		return result;
	}
};

template <typename _PredecessorElementType, size_t _PredecessorDimensionality, typename _ResultElementType, size_t _ResultDimensionality,
	typename _OperationType>
class Accumulate
	: public HandlerDataFilterWithDescriptionBase<_ResultElementType, _ResultDimensionality,
		  AccumulateHandler<_PredecessorElementType, _PredecessorDimensionality, _ResultElementType, _ResultDimensionality, _OperationType>>
{
public:
	static const size_t ResultDimensionality = _ResultDimensionality;

	using AccumulateOperationType = _OperationType;
	using HandlerType =
		AccumulateHandler<_PredecessorElementType, _PredecessorDimensionality, _ResultElementType, _ResultDimensionality, _OperationType>;

	Accumulate(AccumulateOperationType operation, ndim::Indices<ResultDimensionality> selectedDimensions, const QString &description)
		: HandlerDataFilterWithDescriptionBase<_ResultElementType, _ResultDimensionality, HandlerType>(
			  HandlerType(operation, selectedDimensions, description))
	{
	}

	ndim::Indices<ResultDimensionality> selectedDimensions() const
	{
		return this->m_handler.unguarded().selectedDimensions;
	}
	void setSelectedDimensions(ndim::Indices<ResultDimensionality> selectedDimensions)
	{
		if (this->m_handler.unguarded().selectedDimensions == selectedDimensions)
			return;
		this->invalidate();
		this->m_handler.lock()->selectedDimensions = selectedDimensions;
	}

	AccumulateOperationType accumulateOperation() const
	{
		return this->m_handler.unguarded().operation;
	}
	void setAccumulateOperation(AccumulateOperationType accumulateOperation)
	{
		this->invalidate();
		this->m_handler.lock()->operation = accumulateOperation;
	}
};

template <typename PredecessorType, size_t ResultDimensionality, typename OperationType>
std::shared_ptr<Accumulate<ElementTypeOf_t<PredecessorType>, DimensionalityOf_t<PredecessorType>::value, ElementTypeOf_t<PredecessorType>,
	ResultDimensionality, OperationType>>
makeAccumulate(std::shared_ptr<PredecessorType> predecessor, OperationType operation, ndim::Indices<ResultDimensionality> selectedDimensions,
	const QString &description)
{
	auto filter = std::make_shared<Accumulate<ElementTypeOf_t<PredecessorType>, DimensionalityOf_t<PredecessorType>::value,
		ElementTypeOf_t<PredecessorType>, ResultDimensionality, OperationType>>(operation, selectedDimensions, description);
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_ACCUMULATE_H
