#ifndef PYFC_SKIPABLE2D_H
#define PYFC_SKIPABLE2D_H

#include "pyfc/filter2d.h"

class Skipable2d : public Filter2d
{
public:
	Skipable2d() = default;

	virtual std::shared_ptr<fc::SkipableDataFilter<float, 2>> getSkipableFilter() const = 0;
};

#endif // PYFC_SKIPABLE2D_H
