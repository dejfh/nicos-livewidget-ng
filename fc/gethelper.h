#ifndef FILTER_GETHELPER_H
#define FILTER_GETHELPER_H

#include "ndim/pointer.h"
#include "fc/datafilter.h"

#include "variadic.h"

#include "ndim/algorithm_omp.h"

namespace fc
{

template <typename ElementType, size_t Dimensionality>
void getData(ValidationProgress &progress, const std::shared_ptr<const DataFilter<ElementType, Dimensionality>> &predecessor,
	ndim::pointer<ElementType, Dimensionality> data)
{
	ndim::Container<ElementType, Dimensionality> container;
	container = data;
	container = predecessor->getData(progress, &container);
	if (data.data != container.constData().data) {
#pragma omp parallel
		{
			ndim::copy_omp(container.constData(), data);
		}
	}
}

template <typename ElementType, size_t Dimensionality>
ndim::Container<ElementType, Dimensionality> getMutableData(ValidationProgress &progress,
	const std::shared_ptr<const DataFilter<ElementType, Dimensionality>> &predecessor,
	ndim::Container<ElementType, Dimensionality> *recycle = nullptr)
{
	auto data = predecessor->getData(progress, recycle);
	if (data.isMutable())
		return data;
	auto copy = ndim::makeMutableContainer(data.layout().sizes, recycle);
#pragma omp parallel
	{
		ndim::copy_omp(data, copy);
	}
	return copy;
}

template <typename ElementType>
void getData(ValidationProgress &progress, const std::shared_ptr<const DataFilter<ElementType>> &predecessor, ElementType &data)
{
	ndim::Container<ElementType> container(ndim::make_pointer(data));
	container = predecessor->getData(progress, &container);
	if (container.constData().data != &data) {
		if (container.isMutable())
			data = std::move(container.mutableData().first());
		else
			data = container.constData().first();
	}
}

} // namespace fc

#endif // FILTER_GETHELPER_H
