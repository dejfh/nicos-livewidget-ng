#ifndef FILTER_DATAFILTERBASE_H
#define FILTER_DATAFILTERBASE_H

#include <tuple>

#include "filter/filterbase.h"

namespace filter
{

template <typename... PredecessorTypes>
class DataFilterHandlerBase : public FilterHandlerBase<PredecessorTypes...>
{
public:
	using SizesTuple = std::tuple<typename PredecessorTypes::SizesType...>;
	using BaseType = FilterHandlerBase<PredecessorTypes...>;
	using PredecessorTuple = typename BaseType::PredecessorTuple;

	using ContainerTuple = std::tuple<Container<typename PredecessorTypes::ElementType, PredecessorTypes::Dimensionality>...>;

private:
	class PrepareFunctor
	{
	private:
		PreparationProgress &m_progress;

	public:
		PrepareFunctor(PreparationProgress &progress)
			: m_progress(progress)
		{
		}

		template <typename ElementType, size_t Dimensionality>
		void operator()(const std::shared_ptr<const DataFilter<ElementType, Dimensionality>> &predecessor, ndim::Sizes<Dimensionality> &sizes) const
		{
			sizes = hlp::notNull(predecessor)->prepare(m_progress);
		}
	};

	template <typename RecycleType>
	class GetDataFunctor
	{
	private:
		ValidationProgress &m_progress;
		RecycleType *m_recycle;

	public:
		GetDataFunctor(ValidationProgress &progress, RecycleType *recycle)
			: m_progress(progress)
			, m_recycle(recycle)
		{
		}

		template <typename ElementType, size_t Dimensionality>
		void operator()(
			const std::shared_ptr<const DataFilter<ElementType, Dimensionality>> &predecessor, Container<ElementType, Dimensionality> &result) const
		{
			Container<ElementType, Dimensionality> *recycle = nullptr;
			hlp::assignIfAssignable(recycle, m_recycle);
			result = predecessor->getData(m_progress, recycle);
		}
	};

protected:
	SizesTuple preparePredecessors(PreparationProgress &progress) const
	{
		SizesTuple sizes;
		hlp::variadic::forEachInTuple(PrepareFunctor(progress), this->predecessors, sizes);
		return sizes;
	}

	template <typename RecycleType = void>
	ContainerTuple getPredecessorsData(ValidationProgress &progress, RecycleType *recycle = nullptr) const
	{
		ContainerTuple container;
		hlp::variadic::forEachInTuple(GetDataFunctor<RecycleType>(progress, recycle), this->predecessors, container);
		return container;
	}
};

template <typename _ResultElementType, size_t _ResultDimensionality, typename _DataOperationType, typename... PredecessorTypes>
class DataFilterHandlerWithOperationBase : public DataFilterHandlerBase<PredecessorTypes...>
{
	/*
	concept _DataOperationType {
		Container<ResultElementType, ResultDimensionality> operator()(Container<typename PredecessorTypes::ElementType,
	PredecessorTypes::Dimensionality>... input) const;
	};
	*/

public:
	using ResultElementType = _ResultElementType;
	static const size_t ResultDimensionality = _ResultDimensionality;
	using DataOperationType = _DataOperationType;
	using ResultContainerType = Container<ResultElementType, ResultDimensionality>;

public:
	DataOperationType dataOperation;
	QString description;

public:
	DataFilterHandlerWithOperationBase(DataOperationType dataOperation = DataOperationType(), const QString &description = QString())
		: dataOperation(dataOperation)
		, description(description)
	{
	}

	ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const
	{
		auto sizes = this->preparePredecessors(progress);
		progress.addStep(ndim::totalCount(std::get<0>(sizes)), description);
		return std::get<0>(sizes);
	}

	Container<ResultElementType, ResultDimensionality> getData(
		ValidationProgress &progress, Container<ResultElementType, ResultDimensionality> *recycle) const
	{
		auto container = this->getPredecessorsData(progress, recycle);
		auto args = std::tuple_cat(std::make_tuple(recycle), std::move(container));
		Container<ResultElementType, ResultDimensionality> result =
			hlp::variadic::_callR(dataOperation, std::move(args), hlp::variadic::makeSequence(args));
		progress.advanceProgress(result.layout().size());
		progress.advanceStep();
		return result;
	}
};

template <typename _ResultElementType, size_t _ResultDimensionality, typename _HandlerType>
class HandlerDataFilterBase : public HandlerFilterBase<_HandlerType>, public virtual DataFilter<_ResultElementType, _ResultDimensionality>
{
	/*
	concept _HandlerType {
		ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const;
		Container<ResultElementType, ResultDimensionality> getData(ValidationProgress &progress, Container<ResultElementType, ResultDimensionality>
	*recycle) const;
	};
	*/

public:
	using ResultElementType = _ResultElementType;
	static const size_t ResultDimensionality = _ResultDimensionality;
	using HandlerType = _HandlerType;

	// DataFilter interface
public:
	template <typename... Args>
	HandlerDataFilterBase(Args &&... handlerArgs)
		: HandlerFilterBase<HandlerType>(std::forward<Args>(handlerArgs)...)
	{
	}

	virtual ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const override
	{
		HandlerType operation = this->m_handler.get();
		progress.throwIfCancelled();

		return operation.prepare(progress);
	}
	virtual Container<ResultElementType, ResultDimensionality> getData(
		ValidationProgress &progress, Container<ResultElementType, ResultDimensionality> *recycle) const override
	{
		HandlerType operation = this->m_handler.get();
		progress.throwIfCancelled();

		return operation.getData(progress, recycle);
	}
};

template <typename _ResultElementType, size_t _ResultDimensionality, typename _HandlerType>
class HandlerDataFilterWithDescriptionBase : public HandlerDataFilterBase<_ResultElementType, _ResultDimensionality, _HandlerType>
{
	/*
	concept _HandlerType {
		QString description;

		ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const;
		Container<ResultElementType, ResultDimensionality> getData(ValidationProgress &progress, Container<ResultElementType, ResultDimensionality>
	*recycle) const;
	};
	*/

public:
	using HandlerType = _HandlerType;

public:
	template <typename... Args>
	HandlerDataFilterWithDescriptionBase(Args &&... handlerArgs)
		: HandlerDataFilterBase<_ResultElementType, _ResultDimensionality, HandlerType>(std::forward<Args>(handlerArgs)...)
	{
	}

	QString description() const
	{
		return this->m_handler.unguarded().description;
	}
	void setDescription(const QString &description)
	{
		if (this->m_handler.unguarded().description == description)
			return;
		this->invalidate();
		this->m_handler.lock()->description = description;
	}
};

} // namespace filter

#endif // FILTER_DATAFILTERBASE_H
