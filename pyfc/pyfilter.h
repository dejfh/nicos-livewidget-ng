#ifndef PYFC_PYFILTER_H
#define PYFC_PYFILTER_H

#include <memory>
#include <fc/datafilter.h>
#include <pyfc/shared_ptr.h>

namespace pyfc
{

template <typename ElementType>
using FilterVar = InheritPtr<const fc::DataFilterVar<ElementType>>;

template <typename ElementType, size_t Dimensionality>
using FilterNd = InheritPtr<const fc::DataFilter<ElementType, Dimensionality>, FilterVar<ElementType>>;

using Validatable = InheritPtr<const fc::Validatable>;

} // namespace pyfc

using FilterVar = pyfc::FilterVar<float>;

using Filter1d = pyfc::FilterNd<float, 1>;
using Filter2d = pyfc::FilterNd<float, 2>;
using Filter3d = pyfc::FilterNd<float, 3>;
using Filter4d = pyfc::FilterNd<float, 4>;

using Validatable = pyfc::Validatable;

#endif // PYFC_PYFILTER_H
