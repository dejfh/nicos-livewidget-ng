#ifndef FILTER_DATAFILTER_H
#define FILTER_DATAFILTER_H

#include "filter/filter.h"

namespace filter
{

template <typename _ElementType, size_t _Dimensionality = 0>
class DataFilter : public virtual Predecessor
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

	using SizesType = ndim::Sizes<Dimensionality>;

	/*!
	 * \brief prepare
	 * \param progress Collects validation steps and notifies about cancelation.
	 * \return Returns the sizes in each dimension of the returned data.
	 */
	virtual ndim::sizes<_Dimensionality> prepare(PreparationProgress &progress) const = 0;
	/*!
	 * \brief getData
	 * \param progress Collects progress information and notifies about cancelation.
	 * \param result The container to receive the data. If the container size does not match the needed size, the container will be resized.
	 */
	virtual Container<_ElementType, _Dimensionality> getData(
		ValidationProgress &progress, Container<_ElementType, _Dimensionality> *recycle = nullptr) const = 0;
};

template <typename ElementType, size_t Dimensionality>
DataFilter<ElementType, Dimensionality> &asDataFilter(DataFilter<ElementType, Dimensionality> &filter)
{
	return filter;
}
template <typename ElementType, size_t Dimensionality>
const DataFilter<ElementType, Dimensionality> &asDataFilter(const DataFilter<ElementType, Dimensionality> &filter)
{
	return filter;
}

template <typename FilterType>
struct AsDataFilter {
	using type = typename std::remove_reference<decltype(asDataFilter(std::declval<FilterType>()))>::type;
};
template <typename FilterType>
using AsDataFilter_t = typename AsDataFilter<FilterType>::type;

template <typename FilterType>
std::shared_ptr<AsDataFilter_t<FilterType>> asDataFilter(std::shared_ptr<FilterType> filter)
{
	return filter;
}

template <typename ElementType, size_t... Dimensionalities>
ElementType _getElementType(const filter::DataFilter<ElementType, Dimensionalities> &...);
template <typename... FilterTypes>
struct ElementTypeOf {
	using type = decltype(_getElementType(std::declval<FilterTypes>()...));
};
template <typename... FilterTypes>
using ElementTypeOf_t = typename ElementTypeOf<FilterTypes...>::type;

template <size_t Dimensionality, typename... ElementTypes>
std::integral_constant<size_t, Dimensionality> _getDimensionality(const filter::DataFilter<ElementTypes, Dimensionality> &...);
template <typename... FilterTypes>
struct DimensionalityOf {
	using type = decltype(_getDimensionality(std::declval<FilterTypes>()...));
};
template <typename... FilterTypes>
using DimensionalityOf_t = typename DimensionalityOf<FilterTypes...>::type;

} // namespace filter

#endif // FILTER_DATAFILTER_H
