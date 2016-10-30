#ifndef PYFC_NOOPFILTER2D_H
#define PYFC_NOOPFILTER2D_H

#include "pyfc/skipable2d.h"

class NoopFilter2d : public Skipable2d
{
public:
	NoopFilter2d() = default;

	// Filter2d interface
public:
	virtual std::shared_ptr<const fc::DataFilter<float, 2>> getFilter() const override;

	// Skipable2d interface
public:
	virtual std::shared_ptr<fc::SkipableDataFilter<float, 2>> getSkipableFilter() const override;
};

inline std::shared_ptr<const fc::DataFilter<float, 2>> NoopFilter2d::getFilter() const
{
	return std::shared_ptr<const fc::DataFilter<float, 2>>(nullptr);
}

inline std::shared_ptr<fc::SkipableDataFilter<float, 2>> NoopFilter2d::getSkipableFilter() const
{
	return std::shared_ptr<fc::SkipableDataFilter<float, 2>>(nullptr);
}

#endif // PYFC_NOOPFILTER2D_H
