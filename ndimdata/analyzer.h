#ifndef NDIMDATA_ANALYZER_H
#define NDIMDATA_ANALYZER_H

#include <cstddef>

#include "ndimdata/statistic.h"

#include "ndim/pointer.h"
#include "helper/asyncprogress.h"

namespace ndimdata
{

size_t analyzeDuration(size_t size);

template <typename _T>
void analyze(_T *data, size_t size, DataStatistic &statistic, double roiRatio, double displayRatio, size_t binCount, hlp::AsyncProgress<size_t> &progress);

} // namespace ndimdata

#endif // NDIMDATA_ANALYZER_H
