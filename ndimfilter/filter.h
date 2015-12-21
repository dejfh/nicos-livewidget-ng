#ifndef NDIMFILTER_FILTER_H
#define NDIMFILTER_FILTER_H

#include <array>

#include "filter/filter.h"
#include "filter/filterbase.h"

#include "ndim/pointer.h"
#include "ndim/buffer.h"
#include "ndim/algorithm_omp.h"

namespace filter
{

// template <typename _ElementType, size_t _Dimensionality>
// struct NDimMeta {
//	using ElementType = _ElementType;
//	static const size_t Dimensionality = _Dimensionality;

//	ndim::sizes<Dimensionality> sizes;
//};

// template <typename _ElementType, size_t _Dimensionality>
// struct NDimFilterTypeTraits {
//	using ElementType = _ElementType;
//	static const size_t Dimensionality = _Dimensionality;
//	using CopyType = ndim::pointer<ElementType, Dimensionality>;
//	using ConstType = ndim::pointer<const ElementType, Dimensionality> &;
//	using MetaType = NDimMeta<ElementType, Dimensionality>;
//	using StoreType = ndim::Buffer<ElementType, Dimensionality>;

//	static void initStore(StoreType &store, const MetaType &meta)
//	{
//		store.resize(meta.sizes);
//	}
//	static CopyType getStoreCopyRef(StoreType &store)
//	{
//		return store.pointer();
//	}
//	static void copyFromStore(const StoreType &store, CopyType dest)
//	{
//#pragma omp parallel
//		{
//			ndim::copy_omp(store.cpointer(), dest);
//		}
//	}
//	static void constFromStore(const StoreType &store, ConstType dest)
//	{
//		dest = store.cpointer();
//	}
//};

// template <typename _ElementType, size_t _Dimensionality>
// using NDimDataFilter = filter::DataFilter<NDimFilterTypeTraits<_ElementType, _Dimensionality>>;

// template <typename _FilterType>
// struct ElementTypeOf {
//	template <typename _ElementType, size_t _Dimensionality>
//	static _ElementType getElementType(const filter::DataFilter<NDimFilterTypeTraits<_ElementType, _Dimensionality>> &);

//	using type = decltype(getElementType(std::declval<_FilterType>()));
//};
// template <typename _FilterType>
// struct DimensionalityOf {
//	template <typename _ElementType, size_t _Dimensionality>
//	static std::integral_constant<size_t, _Dimensionality> getElementType(const filter::DataFilter<NDimFilterTypeTraits<_ElementType,
//_Dimensionality>> &);

//	using type = decltype(getElementType(std::declval<_FilterType>()));
//};

// template <typename _FilterType>
// using ElementTypeOf_t = typename ElementTypeOf<_FilterType>::type;
// template <typename _FilterType>
// using DimensionalityOf_t = typename DimensionalityOf<_FilterType>::type;

} // namespace ndimfilter

#endif // NDIMFILTER_FILTER_H
