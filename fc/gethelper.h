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
	Container<ElementType, Dimensionality> container;
	container.setMutablePointer(data);
	container = predecessor->getData(progress, &container);
	if (data.data != container.constData().data) {
#pragma omp parallel
		{
			ndim::copy_omp(container.constData(), data);
		}
	}
}

template <typename ElementType, size_t Dimensionality>
Container<ElementType, Dimensionality> getMutableData(ValidationProgress &progress,
	const std::shared_ptr<const DataFilter<ElementType, Dimensionality>> &predecessor, Container<ElementType, Dimensionality> *recycle = nullptr)
{
	auto data = predecessor->getData(progress, recycle);
	if (data.isMutable())
		return data;
	auto copy = fc::makeMutableContainer(data.layout().sizes, recycle);
#pragma omp parallel
	{
		ndim::copy_omp(data, copy);
	}
	return copy;
}

template <typename ElementType>
void getData(ValidationProgress &progress, const std::shared_ptr<const DataFilter<ElementType>> &predecessor, ElementType &data)
{
	Container<ElementType> container;
	container.setMutablePointer(ndim::make_pointer(&data, ndim::Sizes<0>()));
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
