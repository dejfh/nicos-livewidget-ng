#ifndef NDIMFILTER_TRANSFORM_H
#define NDIMFILTER_TRANSFORM_H

#include <tuple>
#include <memory>
#include <array>

#include "filter/filter.h"
#include "filter/filterbase.h"
#include "filter/gethelper.h"
#include "filter/typetraits.h"

#include "ndim/pointer.h"
#include "ndim/algorithm_omp.h"

#include "helper/variadic.h"

#include "safecast.h"

namespace filter
{

class PredecessorStoreSet
{
	const std::shared_ptr<filter::Successor> successor;

public:
	explicit PredecessorStoreSet(std::shared_ptr<filter::Successor> successor)
		: successor(std::move(successor))
	{
	}

	template <typename _Predecessor>
	void operator()(filter::PredecessorStore<_Predecessor> &store, std::shared_ptr<const _Predecessor> predecessor) const
	{
		store.reset(std::move(predecessor), successor);
	}
};

class PredecessorStoreClear
{
	filter::Successor *const successor;

public:
	explicit PredecessorStoreClear(filter::Successor *successor)
		: successor(successor)
	{
	}

	template <typename _Predecessor>
	void operator()(filter::PredecessorStore<_Predecessor> &store) const
	{
		store.clear(successor);
	}
};

class PredecessorGet
{
public:
	template <typename _Predecessor>
	void operator()(const filter::PredecessorStore<_Predecessor> &store, std::shared_ptr<const _Predecessor> &ptr) const
	{
		ptr = store.get();
	}
};

class PredecessorPrepare
{
	PreparationProgress &progress;

public:
	explicit PredecessorPrepare(PreparationProgress &progress)
		: progress(progress)
	{
	}

	template <typename ElementType, size_t Dimensionality>
	void operator()(
		const std::shared_ptr<const DataFilter<ElementType, Dimensionality>> &predecessor, std::array<size_t, Dimensionality> &sizes) const
	{
		hlp::notNull(predecessor);
		sizes = predecessor->prepareConst(progress);
	}
};

class PredecessorGetData
{
	filter::ValidationProgress &progress;

public:
	PredecessorGetData(filter::ValidationProgress &progress)
		: progress(progress)
	{
	}

	template <typename _ElementType, size_t _Dimensionality>
	void operator()(const std::shared_ptr<const DataFilter<_ElementType, _Dimensionality>> &predecessor,
		Container<const _ElementType, _Dimensionality> &buffer, ndim::pointer<const _ElementType, _Dimensionality> &pointer) const
	{
		buffer = filter::getConstData(progress, predecessor);
		pointer = buffer.pointer();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <typename _OperationType, size_t _Dimensionality, typename... _InputTypes>
class TransformFilter : public FilterBase,
						public virtual filter::NoConstDataFilter<hlp::variadic::result_of_t<_OperationType, _InputTypes...>, _Dimensionality>
{
public:
	static const size_t Dimensionality = _Dimensionality;

private:
	using PredecessorStoreTuple = std::tuple<filter::PredecessorStore<DataFilter<_InputTypes, Dimensionality>>...>;
	using PredecessorTuple = std::tuple<std::shared_ptr<const DataFilter<_InputTypes, Dimensionality>>...>;
	using BufferTuple = std::tuple<Container<const _InputTypes, Dimensionality>...>;
	using PointerTuple = std::tuple<ndim::pointer<const _InputTypes, Dimensionality>...>;

	using OperationType = _OperationType;
	using OutputType = hlp::variadic::result_of_t<OperationType, _InputTypes...>;

	PredecessorStoreTuple m_predecessors;
	OperationType m_operation;
	size_t m_opFlops;

	const QString m_description;

public:
	explicit TransformFilter(_OperationType operation, size_t opFlops, const QString &description)
		: m_operation(operation)
		, m_opFlops(opFlops)
		, m_description(description)
	{
	}
	~TransformFilter()
	{
		hlp::variadic::forEachInTuple(PredecessorStoreClear(this), m_predecessors);
	}

	void setPredecessors(PredecessorTuple predecessors)
	{
		// TODO: unnecessary copies of predecessors?
		hlp::variadic::forEachInTuple(PredecessorStoreSet(this->shared_from_this()), m_predecessors, std::move(predecessors));
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<OutputType, Dimensionality>>) const override
	{
		PredecessorTuple predecessors;
		hlp::variadic::forEachInTuple(PredecessorGet(), m_predecessors, predecessors);
		size_t opFlops = m_opFlops;
		progress.throwIfCancelled();

		std::array<ndim::Sizes<Dimensionality>, sizeof...(_InputTypes)> predecessorSizes;
		hlp::variadic::forEachInTuple(PredecessorPrepare(progress), predecessors, predecessorSizes);
		// TODO: Check predecessor sizes
		progress.addStep(std::get<0>(predecessorSizes).size() * opFlops, m_description);

		return std::get<0>(predecessorSizes);
	}
	virtual void getData(ValidationProgress &progress, Container<OutputType, Dimensionality> &result,
		OverloadDummy<DataFilter<OutputType, Dimensionality>>) const override
	{
		PredecessorTuple predecessors;
		hlp::variadic::forEachInTuple(PredecessorGet(), m_predecessors, predecessors);
		size_t opFlops = m_opFlops;
		OperationType operation = m_operation;
		progress.throwIfCancelled();

		BufferTuple buffers;
		PointerTuple pointers;
		hlp::variadic::forEachInTuple(PredecessorGetData(progress), predecessors, buffers, pointers);

		result.resize(std::get<0>(pointers).sizes);

#pragma omp parallel
		{
			ndim::transform_omp(result.pointer(), operation, pointers);
		}
		progress.advanceProgress(std::get<0>(pointers).size() * opFlops);
		progress.advanceStep();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename _OperationType, typename... _PredecessorTypes>
std::shared_ptr<TransformFilter<_OperationType, DimensionalityOf_t<_PredecessorTypes...>::value, ElementTypeOf_t<_PredecessorTypes>...>>
makeTransform(QString description, _OperationType operation, size_t opFlops, std::shared_ptr<_PredecessorTypes>... inputFilters)
{
	auto filter =
		std::make_shared<TransformFilter<_OperationType, DimensionalityOf_t<_PredecessorTypes...>::value, ElementTypeOf_t<_PredecessorTypes>...>>(
			operation, opFlops, std::move(description));
	filter->setPredecessors(std::make_tuple(inputFilters...));
	return filter;
}

} // namespace ndimfilter

#endif // NDIMFILTER_TRANSFORM_H
