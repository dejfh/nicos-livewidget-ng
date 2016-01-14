#ifndef LW_IMAGEFILTERBASE_H
#define LW_IMAGEFILTERBASE_H

#include "filter/filterbase.h"

#include "lw/imagefilter.h"

namespace lw
{

class ImageFilterBase : public ImageFilter,
						public filter::SinglePredecessorFilterBase<filter::DataFilter<float, 2>>,
						public filter::NoConstDataFilter<float, 2>
{
	bool m_isEnabled;

protected:
	virtual prepareOverride(filter::PreparationProgress &progress, ndim::Sizes<2> sizes) const = 0;

	virtual bool supportsInPlace() const = 0;

	virtual applyFilter(filter::ValidationProgress &progress, ndim::pointer<const float, 2> in, ndim::pointer<float, 2> out) const = 0;

public:
	ImageFilterBase()
		: m_isEnabled(true)
	{
	}

	// DataFilter interface
public:
    virtual ndim::sizes<2> prepare(filter::PreparationProgress &progress) const override
	{
		bool enabled = m_isEnabled;
		progress.throwIfCancelled();
		if (enabled) {
			ndim::sizes<2> sizes = predecessor()->prepareConst(progress);
			prepareOverride(progress, sizes);
			return sizes;
		} else {
			return predecessor()->prepare(progress);
		}
	}
	virtual void getData(
        filter::ValidationProgress &progress, filter::Container<float, 2> *recycle) const override
	{
		bool enabled = m_isEnabled;
		progress.throwIfCancelled();
		if (enabled) {
			filter::Container<const float, 2> buffer;
			predecessor->getConstData();
		} else {
            predecessor()->getData(progress, recycle);
		}
	}

	// DataFilter interface
public:
    virtual ndim::sizes<2> prepareConst(filter::PreparationProgress &progress) const override
	{
		bool enabled = m_isEnabled;
		progress.throwIfCancelled();
		if (enabled) {
		} else {
			predecessor()->prepareConst(progress);
		}
	}
	virtual void getConstData(
        filter::ValidationProgress &progress, filter::Container<const float, 2> *recycle) const override
	{
		bool enabled = m_isEnabled;
		progress.throwIfCancelled();
		if (enabled) {
		} else {
            predecessor()->getConstData(progress, recycle);
		}
	}

	// ImageFilter interface
public:
	virtual std::shared_ptr<filter::DataFilter<float, 2>> predecessor() const override
	{
		return filter::SinglePredecessorFilterBase<filter::DataFilter<float, 2>>::predecessor();
	}
	virtual void setPredecessor(std::shared_ptr<const filter::DataFilter<float, 2>> predecessor) override
	{
		filter::SinglePredecessorFilterBase<filter::DataFilter<float, 2>>::setPredecessor(predecessor);
	}

	virtual bool isEnabled() const override
	{
		return m_isEnabled;
	}
	virtual void setEnabled(bool on) override
	{
		this->setAndInvalidate(m_isEnabled, on);
	}

	virtual bool hasControlWidget() const override = 0;

	virtual std::unique_ptr<QWidget> createControlWidget() override = 0;
};

} // namespace lw

#endif // LW_IMAGEFILTERBASE_H
