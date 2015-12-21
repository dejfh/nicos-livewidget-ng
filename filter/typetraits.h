#ifndef FILTER_TYPETRAITS_H
#define FILTER_TYPETRAITS_H

#include "filter/filter.h"

namespace filter
{

template <typename _ElementType, size_t... _Dimensionalities>
static _ElementType _getElementType(const filter::DataFilter<_ElementType, _Dimensionalities> &...);

template <typename... _FilterTypes>
struct ElementTypeOf {
	using type = decltype(_getElementType(std::declval<_FilterTypes>()...));
};

template <size_t _Dimensionality, typename... _ElementTypes>
static std::integral_constant<size_t, _Dimensionality> _getDimensionality(const filter::DataFilter<_ElementTypes, _Dimensionality> &...);

template <typename... _FilterTypes>
struct DimensionalityOf {
	using type = decltype(_getDimensionality(std::declval<_FilterTypes>()...));
};

template <typename... _FilterTypes>
using ElementTypeOf_t = typename ElementTypeOf<_FilterTypes...>::type;
template <typename... _FilterTypes>
using DimensionalityOf_t = typename DimensionalityOf<_FilterTypes...>::type;

} // namespace filter

#endif // FILTER_TYPETRAITS_H
