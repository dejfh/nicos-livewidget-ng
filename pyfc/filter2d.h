#ifndef PYFC_FILTER2D_H
#define PYFC_FILTER2D_H

#include <fc/datafilter.h>

class Filter2d
{
public:
	Filter2d() = default;
	virtual ~Filter2d() = default;

	virtual std::shared_ptr<const fc::DataFilter<float, 2>> getFilter() const = 0;
};

#endif // PYFC_FILTER2D_H
