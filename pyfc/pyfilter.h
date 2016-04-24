#ifndef PYFC_PYFILTER_H
#define PYFC_PYFILTER_H

#include <fc/datafilter.h>

namespace pyfc {

template <size_t _Dimensionality>
class PyFilterNd
{
public:
    PyFilterNd() = default;
    virtual ~PyFilterNd() = default;

    virtual std::shared_ptr<const fc::DataFilter<float, _Dimensionality>> getFilter() const = 0;
};

} // namespace pyfc

using Filter1d = pyfc::PyFilterNd<1>;
using Filter2d = pyfc::PyFilterNd<2>;
using Filter3d = pyfc::PyFilterNd<3>;
using Filter4d = pyfc::PyFilterNd<4>;

#endif // PYFC_PYFILTER_H
