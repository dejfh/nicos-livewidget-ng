#ifndef NDIM_MEAN_H
#define NDIM_MEAN_H

#include <algorithm>
#include <vector>
#include <numeric>
#include <utility>

#include "ndim/pointer.h"
#include "ndim/layout.h"
#include "ndim/iterator.h"
#include "ndim/strideiterator.h"

#include "asyncprogress.h"

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

namespace ndim
{

template <size_t _D_in, size_t _D_out>
size_t inPlaceMeanDuration(ndim::Sizes<_D_in> sizes, ndim::Indices<_D_out> selectedDimensions, size_t start, size_t end)
{
	ndim::Indices<_D_in - _D_out> reducedDimensions = hlp::array::invertSelection<_D_in>(selectedDimensions);
	ndim::sizes<_D_in - _D_out> reducedSizes = hlp::array::select(sizes, reducedDimensions);
	size_t reduceSize = reducedSizes.size();

	size_t duration = 1;
	if (start > 0)
		++duration;
	if (end < reduceSize)
		++duration;
	return sizes.size() * duration;
}

template <typename _T_in, typename _T_out, size_t _D_in, size_t _D_out>
void inPlaceMean(ndim::pointer<_T_in, _D_in> in, ndim::pointer<_T_out, _D_out> out, std::array<size_t, _D_out> selectedDimensions, size_t start,
	size_t end, AsyncProgress &progress = AsyncProgress())
{
	ndim::Indices<_D_in - _D_out> reducedDimensions = hlp::array::invertSelection<_D_in>(selectedDimensions);
	ndim::pointer<_T_in, _D_in - _D_out> reduced = in.selectDimensions(reducedDimensions);
	size_t reduceSize = reduced.size();

	assert(reduced.isContiguous());
	start = std::max(size_t(0), std::min(start, reduceSize - 1));
	end = std::max(start + 1, std::min(end, reduceSize));

	ndim::pointer<_T_in, _D_out> selected = in.selectDimensions(selectedDimensions);
	size_t selectedSize = selected.size();

#pragma omp parallel
	{
		ndim::iterator<_T_in, _D_out> itIn(selected);
		ndim::iterator<_T_out, _D_out> itOut(out);

#ifdef _OPENMP
		int threadIndex = omp_get_thread_num();
		int threadCount = omp_get_num_threads();
		size_t ompStart = selectedSize * threadIndex / threadCount;
		size_t ompEnd = selectedSize * (threadIndex + 1) / threadCount;
		size_t count = ompEnd - ompStart;

		itIn += ompStart;
		itOut += ompStart;
#else  // _OPENMP
		size_t count = selectedSize;
#endif // _OPENMP

		for (; count > 0; --count, ++itIn, ++itOut) {
			_T_in *itBegin = &(*itIn);
			_T_in *itEnd = itBegin + reduceSize;
			itEnd = itBegin + reduceSize;
			if (start > 0)
				std::nth_element(itBegin, itBegin + start, itEnd);
			if (end < reduceSize)
				std::nth_element(itBegin + start, itBegin + end, itEnd);
			*itOut = std::accumulate(itBegin + start, itBegin + end, _T_out()) / _T_out(end - start + 1);
		}
	}

	progress.advanceProgress(inPlaceMeanDuration(in.sizes, selectedDimensions, start, end));
}

} // namespace ndim

#endif // NDIM_MEAN_H
