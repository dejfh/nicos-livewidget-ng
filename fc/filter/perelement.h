#ifndef FC_FILTER_PERELEMENT_H
#define FC_FILTER_PERELEMENT_H

#include <tuple>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "ndim/algorithm_omp.h"

namespace fc
{
namespace filter
{

template <typename ResutElementType, size_t Dimensionality, typename ElementOperationType, typename... PredecessorElementTypes>
struct PerElementContainerOperation {
	ElementOperationType m_elementOperation;

	PerElementContainerOperation(ElementOperationType elementOperation = ElementOperationType())
		: m_elementOperation(elementOperation)
	{
	}

	ndim::Container<ResutElementType, Dimensionality> operator()(
		ndim::Container<ResutElementType, Dimensionality> *recycle, ndim::Container<PredecessorElementTypes, Dimensionality>... inputs) const
	{
		using ContainerType = ndim::Container<typename std::result_of<ElementOperationType(PredecessorElementTypes...)>::type, Dimensionality>;
		auto &input0 = std::get<0>(std::tie(inputs...));
		ContainerType result = ndim::makeMutableContainer(input0.layout().sizes, recycle);
#pragma omp parallel
		{
			ndim::transform_omp(result.mutableData(), m_elementOperation, std::make_tuple(inputs.constData()...));
		}
		return result;
	}
};

template <typename ElementOperationType, typename... PredecessorTypes>
struct _MakePerElementHelper {
	static const size_t Dimensionality = DimensionalityOf_t<PredecessorTypes...>::value;
	using ResultElementType = typename std::result_of<ElementOperationType(ElementTypeOf_t<PredecessorTypes>...)>::type;

	using ContainerOperationType =
		PerElementContainerOperation<ResultElementType, Dimensionality, ElementOperationType, ElementTypeOf_t<PredecessorTypes>...>;

	using HandlerType =
		DataFilterHandlerWithOperationBase<ResultElementType, Dimensionality, ContainerOperationType, AsDataFilter_t<PredecessorTypes>...>;

	using FilterType = HandlerDataFilterWithDescriptionBase<ResultElementType, Dimensionality, HandlerType>;

	static std::shared_ptr<FilterType> makeFilter(
		const QString &description, ElementOperationType operation, std::shared_ptr<PredecessorTypes>... predecessors)
	{
		auto filter = std::make_shared<FilterType>(operation, description);
		filter->setPredecessors(std::make_tuple(std::move(predecessors)...));
		return filter;
	}
};

template <typename ElementOperationType, typename... PredecessorTypes>
std::shared_ptr<typename _MakePerElementHelper<ElementOperationType, PredecessorTypes...>::FilterType> makePerElement(
	QString description, ElementOperationType elementOperation, std::shared_ptr<PredecessorTypes>... predecessors)
{
	return _MakePerElementHelper<ElementOperationType, PredecessorTypes...>::makeFilter(description, elementOperation, std::move(predecessors)...);
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_PERELEMENT_H
