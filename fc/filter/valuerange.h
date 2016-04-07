#ifndef FC_FILTER_VALUERANGE_H
#define FC_FILTER_VALUERANGE_H

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "fc/gethelper.h"

#include "ndimdata/statistic.h"

namespace fc
{
namespace filter
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

	ndim::Container<std::pair<double, double>> getData(ValidationProgress &progress, ndim::Container<std::pair<double, double>> *recycle) const
	{
		progress.throwIfCancelled();

		std::pair<double, double> range = this->range;
		if (!isSet) {
			ndimdata::DataStatistic stat;
			fc::getData(progress, this->predecessor(), stat);

			if (useFullRange)
				range = std::make_pair(stat.min, stat.max);
			else
				range = std::make_pair(stat.auto_low_bound, stat.auto_high_bound);
		}

		if (invert)
			std::swap(range.first, range.second);

		auto result = ndim::makeMutableContainer(ndim::Sizes<0>(), recycle);
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

} // namespace filter
} // namespace fc

#endif // FC_FILTER_VALUERANGE_H
