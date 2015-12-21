#ifndef FILTER_CHAINS_ZPLOT_H
#define FILTER_CHAINS_ZPLOT_H

#include <memory>

#include "filter/filter.h"

#include "filter/chains/fitspile.h"
#include "ndimfilter/range.h"
#include "ndimfilter/extend.h"
#include "filter/buffer.h"
#include "ndimfilter/accumulate.h"

namespace filter
{
namespace chains
{

class ZPlotChain
{
	FitsPileChain<float, 2> m_fitsPileChain;

	std::shared_ptr<filter::range<float, 2>> m_diRange;
	std::shared_ptr<filter::range<float, 2>> m_obRange;

	std::shared_ptr<filter::extend<float, 2, 3>> m_diExtend;
	std::shared_ptr<filter::extend<float, 2, 3>> m_obExtend;

	std::shared_ptr<filter::Buffer<float, 1>> m_buffer;

public:
	ZPlotChain(std::shared_ptr<const DataFilter<float, 2>> darkImage, std::shared_ptr<const DataFilter<float, 2>> openBeam)
	{
		m_diRange = filter::makeRange(darkImage);
		m_obRange = filter::makeRange(openBeam);

		m_diExtend = filter::makeExtend(m_diRange, ndim::Indices<1>{0}, ndim::Sizes<1>{0});
		m_obExtend = filter::makeExtend(m_obRange, ndim::Indices<1>{0}, ndim::Sizes<1>{0});

		auto normalizeOperation = [](float image, float darkImage, float openBeam) { return (image - darkImage) / openBeam; };
		auto normalized =
			filter::makeTransform("Normalizing z-stack...", normalizeOperation, 2, m_fitsPileChain.pileBuffer(), m_diExtend, m_obExtend);

		auto accumulate = filter::makeAccumulate(normalized, std::plus<float>(), ndim::Indices<1>{0}, "Accumulating z-stack...");

		m_buffer = filter::makeBuffer(accumulate);
	}

	void setFilenames(const QStringList &filenames)
	{
		m_fitsPileChain.setFilenames(filenames);
		m_diExtend->setExtendSizes(std::array<size_t, 1>{size_t(filenames.size())});
		m_obExtend->setExtendSizes(std::array<size_t, 1>{size_t(filenames.size())});
	}

	void setRange(ndim::range<2> range)
	{
		m_fitsPileChain.setRange(range);
		m_diRange->setRange(range);
		m_obRange->setRange(range);
	}

	std::shared_ptr<const filter::Buffer<float, 1>> stackBuffer() const
	{
		return m_buffer;
	}
};

} // namespace chains
} // namespace filter

#endif // FILTER_CHAINS_ZPLOT_H
