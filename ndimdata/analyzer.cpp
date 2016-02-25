#include "ndimdata/analyzer.h"

#include <utility>

#include "ndim/iterator.h"

namespace ndimdata
{

size_t analyzeDuration(size_t size)
{
	return 4 * size;
}

template <typename _T>
std::vector<size_t> build_histogram(_T *data, size_t size, size_t countNan, size_t binCount, _T min, _T max)
{
	std::vector<size_t> histogram(binCount);
	auto begin = data + countNan;
	auto end = data + size;
	double bin_width = (max - min) / binCount * (1.0 + std::numeric_limits<double>::epsilon());
	if (bin_width == 0.0)
		bin_width = 1.0;
	// * (1 + epsilon) ensures not to overflow at max element
	for (auto it = begin; it != end; ++it) {
		_T v = *it;
		int index = int((v - min) / bin_width);
		++histogram[index];
	}
	return std::move(histogram);
}

template <typename _T>
_T get_bound(_T *data, size_t size, size_t count_nan, double ratio)
{
	auto begin = data + count_nan;
	auto end = data + size;
	auto nth = begin + ptrdiff_t((end - begin) * ratio);
	std::nth_element(begin, nth, end);
	if (nth == end)
		--nth;
	return *nth;
}

template <typename _T>
void analyze(_T *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress)
{
	_T min = std::numeric_limits<_T>::max();
	_T max = std::numeric_limits<_T>::lowest();

	statistic.count = statistic.count_nan = 0;
	statistic.sum = statistic.sum_squares = 0.0;

	for (auto it = data, end = data + size; it != end; ++it) {
		_T v = *it;
		if (std::isfinite(double(v))) {
			++statistic.count;
			min = std::min(min, v);
			max = std::max(max, v);
			statistic.sum += v;
			statistic.sum_squares += v * v;
		} else {
			if (statistic.count_nan != size_t(it - data))
				std::swap(*it, data[statistic.count_nan]);
			++statistic.count_nan;
		}
	}

	statistic.min = double(min);
	statistic.max = double(max);

	{
		double lowRoiRatio = (1 - roiRatio) / 2;
		double marginRatio = (1 - displayRatio) / 2;

		double roiLow = double(get_bound(data, size, statistic.count_nan, lowRoiRatio));
		double roiHigh = double(get_bound(data, size, statistic.count_nan, 1 - lowRoiRatio));
		double roiRange = roiHigh - roiLow;
		double displayRange = roiRange / displayRatio;
		double margin = displayRange * marginRatio;

		statistic.auto_low_bound = std::max(statistic.min, roiLow - margin);
		statistic.auto_high_bound = std::min(statistic.max, statistic.auto_low_bound + displayRange);
		statistic.auto_low_bound = std::max(statistic.min, statistic.auto_high_bound - displayRange);
	}

	statistic.histogram = build_histogram(data, size, statistic.count_nan, binCount, min, max);

	progress.advanceProgress(analyzeDuration(size));
}

template void analyze(signed char *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(signed short *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(signed long *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(signed long long *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(unsigned char *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(unsigned short *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(unsigned long *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(unsigned long long *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(float *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);
template void analyze(double *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, AsyncProgress<size_t> &progress);

} // namespace ndimdata
