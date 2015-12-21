#ifndef NDIMDATA_STATISTIC_H
#define NDIMDATA_STATISTIC_H

#include <cstddef>
#include <vector>
#include <cmath>

namespace ndimdata {

struct DataStatistic {
	double min, max;
	size_t count, count_nan;
	double sum, sum_squares;

	double auto_low_bound, auto_high_bound;

	inline double width() const
	{
		return max - min;
	}
	inline double mean_value() const
	{
		return sum / count;
	}
	inline double variance() const
	{
		return (sum_squares - sum * sum / count) / count;
	}
	inline double standard_deviation() const
	{
		return std::sqrt(variance());
	}

	std::vector<size_t> histogram;
};

} // namespace ndimdata

#endif // NDIMDATA_STATISTIC_H

