#ifndef FILTER_FS_VALUERANGE_H
#define FILTER_FS_VALUERANGE_H

#include "filter/datafilter.h"
#include "filter/datafilterbase.h"

#include "filter/gethelper.h"

#include "ndimdata/statistic.h"

namespace filter
{
namespace fs
{

class ValueRangeHandler : public DataFilterHandlerBase<DataFilter<ndimdata::DataStatistic>>
{
public:
	using ResultElementType = std::pair<double, double>;
	static const size_t Dimensionality = 0;

	std::pair<double, double> range;
	bool isSet;
	bool useFullRange;
	bool invert;

	ndim::sizes<0> prepare(PreparationProgress &progress) const
	{
		progress.throwIfCancelled();
		if (!isSet)
			this->preparePredecessors(progress);
		return ndim::makeSizes();
	}

	Container<std::pair<double, double>> getData(ValidationProgress &progress, Container<std::pair<double, double>> *recycle) const
	{
		progress.throwIfCancelled();

		std::pair<double, double> range = this->range;
		if (!isSet) {
			ndimdata::DataStatistic stat;
			filter::getData(progress, this->predecessor(), stat);

			if (useFullRange)
				range = std::make_pair(stat.min, stat.max);
			else
				range = std::make_pair(stat.auto_low_bound, stat.auto_high_bound);
		}

		if (invert)
			std::swap(range.first, range.second);

		auto result = filter::makeMutableContainer(ndim::Sizes<0>(), recycle);
		result.mutableData().first() = range;

		return result;
	}
};

class ValueRange : public HandlerDataFilterBase<std::pair<double, double>, 0, ValueRangeHandler>
{
public:
	bool isRangeSet() const
	{
		return this->m_handler.unguarded().isSet;
	}
	std::pair<double, double> range() const
	{
		return this->m_handler.unguarded().range;
	}
	void setRange(std::pair<double, double> range)
	{
		auto &unguarded = this->m_handler.unguarded();
		if (unguarded.isSet && unguarded.range == range)
			return;
		this->invalidate();
		auto guard = this->m_handler.lock();
		guard->isSet = true;
		guard->range = range;
	}
	void setAuto(bool useFullRange)
	{
		auto &unguarded = this->m_handler.unguarded();
		if (!unguarded.isSet && unguarded.useFullRange == useFullRange)
			return;
		this->invalidate();
		auto guard = this->m_handler.lock();
		guard->isSet = false;
		guard->useFullRange = useFullRange;
	}

	bool invert() const
	{
		return this->m_handler.unguarded().isSet;
	}
	void setInvert(bool invert)
	{
		if (this->m_handler.unguarded().invert == invert)
			return;
		this->invalidate();
		this->m_handler.lock()->invert = invert;
	}
};

inline std::shared_ptr<ValueRange> makeValueRange(std::shared_ptr<const DataFilter<ndimdata::DataStatistic>> predecessor)
{
	auto filter = std::make_shared<ValueRange>();
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace fs
} // namespace filter

#endif // FILTER_FS_VALUERANGE_H
