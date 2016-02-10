#ifndef PYFC_SKIPABLE2D_H
#define PYFC_SKIPABLE2D_H

#include "pyfc/filter2d.h"

namespace pyfc
{

class Skipable2d : public Filter2d
{
public:
	Skipable2d() = default;
	~Skipable2d() = default;

	virtual std::shared_ptr<const fc::DataFilter<float, 2>> getSkipableFilter() const;
};

} // namespace pyfc

#endif // PYFC_SKIPABLE2D_H
