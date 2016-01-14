#ifndef FILTER_FS_PERELEMENT_H
#define FILTER_FS_PERELEMENT_H

#include <tuple>

#include "filter/datafilter.h"
#include "filter/datafilterbase.h"

#include "ndim/algorithm_omp.h"

namespace filter
{
namespace fs
{

template <typename ResutElementType, size_t Dimensionality, typename ElementOperationType, typename... PredecessorElementTypes>
struct PerElementContainerOperation {
	ElementOperationType m_elementOperation;

	PerElementContainerOperation(ElementOperationType elementOperation = ElementOperationType())
		: m_elementOperation(elementOperation)
	{
	}

	Container<ResutElementType, Dimensionality> operator()(
		Container<ResutElementType, Dimensionality> *recycle, Container<PredecessorElementTypes, Dimensionality>... inputs) const
	{
		using ContainerType = Container<typename std::result_of<ElementOperationType(PredecessorElementTypes...)>::type, Dimensionality>;
		auto &input0 = std::get<0>(std::tie(inputs...));
		ContainerType result = filter::makeMutableContainer(input0.layout().sizes, recycle);
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

} // namespace fs
} // namespace filter

#endif // FILTER_FS_PERELEMENT_H
