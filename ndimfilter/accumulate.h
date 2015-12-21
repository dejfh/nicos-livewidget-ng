#ifndef NDIMFILTER_ACCUMULATE_H
#define NDIMFILTER_ACCUMULATE_H

#include "ndim/algorithm_omp.h"
#include "ndim/buffer.h"
#include "ndim/iterator.h"

#include "filter/filter.h"
#include "filter/filterbase.h"
#include "filter/typetraits.h"
#include "ndimfilter/filter.h"
#include "filter/gethelper.h"

#include "helper/helper.h"

namespace filter
{

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _InputType, size_t _PredecessorDimensionality, size_t _ResultDimensionality, typename _OperationType>
class Accumulate : public filter::SinglePredecessorFilterBase<DataFilter<_InputType, _PredecessorDimensionality>>,
				   public virtual filter::NoConstDataFilter<_InputType, _ResultDimensionality>
{
	using InputType = _InputType;
	static const size_t PredecessorDimensionality = _PredecessorDimensionality;
	static const size_t ResultDimensionality = _ResultDimensionality;
	static const size_t DimensionalityDifference = _PredecessorDimensionality - _ResultDimensionality;
	using OperationType = _OperationType;
	using OutputType = _InputType;
	// TODO: Support different accumulate Type

	const OperationType m_operation;
	std::array<size_t, ResultDimensionality> m_resultDimensions;
	const QString m_description;

public:
	Accumulate(_OperationType operation, std::array<size_t, ResultDimensionality> selectedDimensions, QString description)
		: m_operation(operation)
		, m_resultDimensions(selectedDimensions)
		, m_description(std::move(description))
	{
	}

	// DataFilter interface
public:
	virtual ndim::sizes<ResultDimensionality> prepare(
		PreparationProgress &progress, OverloadDummy<DataFilter<OutputType, ResultDimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto selectedDimensions = m_resultDimensions;
		progress.throwIfCancelled();
		hlp::notNull(predecessor.get());

		auto predecessorSizes = predecessor->prepareConst(progress);

		progress.addStep(predecessorSizes.size(), m_description);
		return predecessorSizes.selectDimensions(selectedDimensions);
	}
	virtual void getData(ValidationProgress &progress, Container<OutputType, ResultDimensionality> &result,
		OverloadDummy<DataFilter<OutputType, ResultDimensionality>>) const override
	{
		auto predecessor = this->predecessor();
		auto selectedDimensions = m_resultDimensions;
		progress.throwIfCancelled();

		auto buffer = filter::getConstData(progress, predecessor);
		ndim::pointer<const InputType, PredecessorDimensionality> bufferPointer = buffer.pointer();

		ndim::pointer<const InputType, ResultDimensionality> selectedPointer = bufferPointer.selectDimensions(selectedDimensions);

		auto accumulateDimensions = hlp::array::invertSelection<PredecessorDimensionality>(selectedDimensions);
		ndim::layout<DimensionalityDifference> accumulateLayout = bufferPointer.getLayout().selectDimensions(accumulateDimensions);

		result.resize(selectedPointer.sizes);
		auto resultPointer = result.pointer();

#pragma omp parallel
		{
			ndim::operate_omp([](OutputType &value) { value = OutputType(); }, resultPointer);
#pragma omp barrier
			auto inIterator = ndim::makeIterator(selectedPointer);
			auto outIterator = ndim::makeIterator(resultPointer);
			for (size_t count = selectedPointer.size(); count > 0; --count, ++inIterator, ++outIterator) {
				ndim::pointer<const InputType, DimensionalityDifference> accPointer(&*inIterator, accumulateLayout);
				_InputType acc = ndim::accumulate_omp(m_operation, OutputType(), accPointer);
#pragma omp critical
				{
					*outIterator = m_operation(*outIterator, acc);
				}
			}
		}

		progress.advanceProgress(accumulateLayout.size());
		progress.advanceStep();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename _Predecessor, size_t _ResultDimensionality, typename _OperationType>
std::shared_ptr<Accumulate<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value, _ResultDimensionality, _OperationType>>
makeAccumulate(std::shared_ptr<_Predecessor> predecessor, _OperationType operation, std::array<size_t, _ResultDimensionality> selectedDimensions,
	const QString &description)
{
	auto filter =
		std::make_shared<Accumulate<ElementTypeOf_t<_Predecessor>, DimensionalityOf_t<_Predecessor>::value, _ResultDimensionality, _OperationType>>(
			operation, selectedDimensions, description);
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace ndimfilter

#endif // NDIMFILTER_ACCUMULATE_H
