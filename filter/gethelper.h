#ifndef FILTER_GETHELPER_H
#define FILTER_GETHELPER_H

#include "ndim/pointer.h"
#include "ndim/buffer.h"
#include "filter/filter.h"

#include "variadic.h"

#include "ndim/algorithm_omp.h"

namespace filter
{

template <typename ElementType, size_t Dimensionality>
Container<const ElementType, Dimensionality> getConstData(
	ValidationProgress &progress, const std::shared_ptr<const DataFilter<ElementType, Dimensionality>> &predecessor)
{
	Container<const ElementType, Dimensionality> container;
	predecessor->getConstData(progress, container);
	return container;
}

template <typename ElementType, size_t Dimensionality>
void getData(ValidationProgress &progress, const std::shared_ptr<const DataFilter<ElementType, Dimensionality>> &predecessor,
	ndim::pointer<ElementType, Dimensionality> data)
{
	Container<ElementType, Dimensionality> container(data);
	predecessor->getData(progress, container);
	if (container.ownsData()) {
#pragma omp parallel
		{
			ndim::copy_omp(container.pointer(), data);
		}
	}
}

// template <typename _FilterTypeTraits>
// bool prepare(
//	const DataFilter<_FilterTypeTraits> *predecessor, AsyncProgress &progress, DurationCounter &counter, typename _FilterTypeTraits::MetaType &meta)
//{
//	bool useConst = predecessor->supportsConst();
//	if (useConst)
//		predecessor->prepareConst(progress, counter, meta);
//	else
//		predecessor->prepare(progress, counter, meta);
//	return useConst;
//}

// template <typename _FilterTypeTraits>
// bool getData(const DataFilter<_FilterTypeTraits> *predecessor, typename _FilterTypeTraits::StoreType &buffer,
//	typename _FilterTypeTraits::ConstType data, const typename _FilterTypeTraits::MetaType &meta, ValidationProgress &progress)
//{
//	bool useConst = predecessor->supportsConst();
//	if (useConst)
//		predecessor->getConstData(progress, data);
//	else {
//		_FilterTypeTraits::initStore(buffer, meta);
//		predecessor->getData(progress, _FilterTypeTraits::getStoreCopyRef(buffer));
//		_FilterTypeTraits::constFromStore(buffer, data);
//	}
//	return useConst;
//}

} // namespace filter

#endif // FILTER_GETHELPER_H
