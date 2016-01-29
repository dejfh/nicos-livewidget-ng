#ifndef FILTER_CHAINS_PIXMAPOUTPUT_H
#define FILTER_CHAINS_PIXMAPOUTPUT_H

#include <memory>

#include "fc/filter.h"
#include "ndimfilter/fits.h"
#include "fc/switch.h"
#include "fc/buffer.h"
#include "ndimfilter/analyzer.h"
#include "ndimfilter/range.h"
#include "ndimfilter/pixmapbuilder.h"
#include "ndimdata/colormap.h"

namespace fc
{
namespace chains
{

template <typename _ElementType>
class ImageOutputChain
{
public:
	using ElementType = _ElementType;

private:
	std::shared_ptr<fc::SwitchControl> m_logSwitch;

	std::shared_ptr<fc::RangeControl<2>> m_regionOfInterest;
	std::shared_ptr<fc::Switch<float, 2>> m_regionOfInterestSwitch;

	std::shared_ptr<fc::Buffer<ndimdata::DataStatistic>> m_statisticBuffer;

	std::shared_ptr<fc::PixmapRange> m_colorRange;
	std::shared_ptr<fc::SwitchControl> m_colormapSwitch;
	std::shared_ptr<fc::Buffer<QPixmap>> m_pixmapBuffer;

public:
	ImageOutputChain(std::shared_ptr<const DataFilter<ElementType, 2>> source)
	{
		// Calculate Log10
		auto imageLog = fc::makeTransform("Applying logarithm...", [](float v) { return std::log10(v); }, 1, source);
		// Make Log skippable
		auto logSwitch = fc::makeSwitch(source, imageLog);

		auto postBuffer = fc::makeBuffer(logSwitch);

		auto region = fc::makeRange(postBuffer);

		auto regionSwitch = fc::makeSwitch(postBuffer, region);

		// Analyze processed data
		auto analyzer = fc::makeAnalyzer(regionSwitch, "Generating statistic...");
		// Buffer statistic
		m_statisticBuffer = fc::makeBuffer(analyzer);
		// Select range for colormaps
		m_colorRange = fc::makePixmapRange(m_statisticBuffer);
		// Apply grayscale colormap
		auto pixmapGrayscale = fc::makePixmapBuilder(postBuffer, m_colorRange, ndimdata::ColorMapGrayscale(0, 1), "Generating image...");
		// Apply color colormap
		auto pixmapColor = fc::makePixmapBuilder(postBuffer, m_colorRange, ndimdata::ColorMapColor(0, 1), "Generating image...");
		// Select final image
		auto colormapSwitch = fc::makeSwitch(pixmapGrayscale, pixmapColor);
		// Buffer final image
		m_pixmapBuffer = fc::makeBuffer(colormapSwitch);

		m_logSwitch = std::move(logSwitch);
		m_regionOfInterest = std::move(region);
		m_regionOfInterestSwitch = std::move(regionSwitch);
		m_colormapSwitch = std::move(colormapSwitch);
	}

	void setLog(bool log)
	{
		m_logSwitch->setSelection(log ? 1 : 0);
	}

	void setColor(bool color)
	{
		m_colormapSwitch->setSelection(color ? 1 : 0);
	}
	void setInvert(bool invert)
	{
		m_colorRange->setInvert(invert);
	}

	void setColormapRange(std::pair<double, double> range)
	{
		m_colorRange->setRange(range);
	}
	void setAutoColormapRange(bool fullRange)
	{
		m_colorRange->setAuto(fullRange);
	}

	void resetRegionOfInterest()
	{
		m_regionOfInterestSwitch->setSelection(0);
	}
	void setRegionOfInterest(ndim::range<2> region)
	{
		m_regionOfInterest->setRange(region);
		m_regionOfInterestSwitch->setSelection(1);
	}

	std::shared_ptr<const fc::DataFilter<float, 2>> roiFilter() const
	{
		return m_regionOfInterestSwitch;
	}

	std::shared_ptr<const fc::Buffer<ndimdata::DataStatistic>> statistic() const
	{
		return m_statisticBuffer;
	}
	std::shared_ptr<const fc::Buffer<QPixmap>> pixmap() const
	{
		return m_pixmapBuffer;
	}
};

} // namespace chains
} // namespace fc

#endif // FILTER_CHAINS_PIXMAPOUTPUT_H
