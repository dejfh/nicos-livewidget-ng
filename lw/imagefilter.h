#ifndef LW_IMAGEFILTER_H
#define LW_IMAGEFILTER_H

#include <memory>

#include <QWidget>

#include "fc/filter.h"

namespace lw
{

class ImageFilter : public fc::DataFilter<float, 2>
{
public:
	virtual std::shared_ptr<fc::DataFilter<float, 2>> predecessor() const = 0;
	virtual void setPredecessor(std::shared_ptr<const fc::DataFilter<float, 2>> predecessor) = 0;

	virtual bool isEnabled() const = 0;
	virtual void setEnabled(bool on) = 0;

	virtual bool hasControlWidget() const = 0;
	virtual std::unique_ptr<QWidget> createControlWidget() = 0;
};

class ImageFilterFactory
{
public:
	virtual std::shared_ptr<ImageFilter> createFilter() = 0;
};

} // namespace lw

#endif // LW_IMAGEFILTER_H
