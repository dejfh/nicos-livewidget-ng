#ifndef PYFC_FILTER2D_H
#define PYFC_FILTER2D_H

#include <fc/datafilter.h>

namespace pyfc
{

class Filter2d
{
public:
	Filter2d() = default;
	~Filter2d() = default;

	virtual std::shared_ptr<const fc::DataFilter<float, 2>> getFilter() const;
};

} // namespace pyfc

#endif // PYFC_FILTER2D_H
