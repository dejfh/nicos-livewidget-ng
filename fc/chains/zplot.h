#ifndef FILTER_CHAINS_ZPLOT_H
#define FILTER_CHAINS_ZPLOT_H

#include <memory>

#include "fc/filter.h"

#include "fc/chains/fitspile.h"
#include "ndimfilter/range.h"
#include "ndimfilter/extend.h"
#include "fc/buffer.h"
#include "ndimfilter/accumulate.h"

namespace fc
{
namespace chains
{

class ZPlotChain
{
	FitsPileChain<float, 2> m_fitsPileChain;

	std::shared_ptr<fc::range<float, 2>> m_diRange;
	std::shared_ptr<fc::range<float, 2>> m_obRange;

	std::shared_ptr<fc::extend<float, 2, 3>> m_diExtend;
	std::shared_ptr<fc::extend<float, 2, 3>> m_obExtend;

	std::shared_ptr<fc::Buffer<float, 1>> m_buffer;

public:
	ZPlotChain(std::shared_ptr<const DataFilter<float, 2>> darkImage, std::shared_ptr<const DataFilter<float, 2>> openBeam)
	{
		m_diRange = fc::makeRange(darkImage);
		m_obRange = fc::makeRange(openBeam);

		m_diExtend = fc::makeExtend(m_diRange, ndim::Indices<1>{0}, ndim::Sizes<1>{0});
		m_obExtend = fc::makeExtend(m_obRange, ndim::Indices<1>{0}, ndim::Sizes<1>{0});

		auto normalizeOperation = [](float image, float darkImage, float openBeam) { return (image - darkImage) / openBeam; };
		auto normalized =
			fc::makeTransform("Normalizing z-stack...", normalizeOperation, 2, m_fitsPileChain.pileBuffer(), m_diExtend, m_obExtend);

		auto accumulate = fc::makeAccumulate(normalized, std::plus<float>(), ndim::Indices<1>{0}, "Accumulating z-stack...");

		m_buffer = fc::makeBuffer(accumulate);
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

	std::shared_ptr<const fc::Buffer<float, 1>> stackBuffer() const
	{
		return m_buffer;
	}
};

} // namespace chains
} // namespace fc

#endif // FILTER_CHAINS_ZPLOT_H
