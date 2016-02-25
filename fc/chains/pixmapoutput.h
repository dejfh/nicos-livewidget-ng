#ifndef FILTER_CHAINS_PIXMAPOUTPUT_H
#define FILTER_CHAINS_PIXMAPOUTPUT_H

#include <memory>

#include "fc/filter.h"
#include "fc/filter/forward.h"
#include "fc/filter/fits.h"
#include "fc/filter/switch.h"
#include "fc/filter/buffer.h"
#include "fc/filter/analyzer.h"
#include "fc/filter/subrange.h"
#include "fc/filter/valuerange.h"
#include "fc/filter/pixmap.h"
#include "fc/filter/perelement.h"

#include "ndimdata/colormap.h"

namespace fc
{
namespace chains
{

class ImageOutputChain
{
private:
	std::shared_ptr<fc::filter::Forward<float, 2>> m_input;

	std::shared_ptr<fc::filter::SwitchControl> m_logSwitch;

	std::shared_ptr<fc::filter::SubrangeControl<2>> m_regionOfInterest;
	std::shared_ptr<fc::filter::Switch<float, 2>> m_regionOfInterestSwitch;

	std::shared_ptr<fc::filter::Buffer<ndimdata::DataStatistic>> m_statisticBuffer;

	std::shared_ptr<fc::filter::ValueRange> m_colorRange;
	std::shared_ptr<fc::filter::SwitchControl> m_colormapSwitch;
	std::shared_ptr<fc::filter::Buffer<QImage>> m_pixmapBuffer;

public:
	ImageOutputChain()
	{
		m_input = std::make_shared<fc::filter::Forward<float, 2>>();

		// Calculate Log10
		auto imageLog = fc::filter::makePerElement("Applying logarithm...", [](float v) { return std::log10(v); }, m_input);
		// Make Log skippable
		auto logSwitch = fc::filter::makeSwitch(m_input, imageLog);

		auto postBuffer = fc::filter::makeBuffer(logSwitch);

		auto region = fc::filter::makeSubrange(postBuffer);

		auto regionSwitch = fc::filter::makeSwitch(postBuffer, region);

		// Analyze processed data
		auto analyzer = fc::filter::makeAnalyzer(regionSwitch, "Generating statistic...");
		// Buffer statistic
		m_statisticBuffer = fc::filter::makeBuffer(analyzer);
		// Select range for colormaps
		m_colorRange = fc::filter::makeValueRange(m_statisticBuffer);
		// Apply grayscale colormap
		auto pixmapGrayscale = fc::filter::makePixmap(postBuffer, m_colorRange, ndimdata::ColorMapGrayscale(0, 1), "Generating image...");
		// Apply color colormap
		auto pixmapColor = fc::filter::makePixmap(postBuffer, m_colorRange, ndimdata::ColorMapColor(0, 1), "Generating image...");
		// Select final image
		auto colormapSwitch = fc::filter::makeSwitch(pixmapGrayscale, pixmapColor);
		// Buffer final image
		m_pixmapBuffer = fc::filter::makeBuffer(colormapSwitch);

		m_logSwitch = std::move(logSwitch);
		m_regionOfInterest = std::move(region);
		m_regionOfInterestSwitch = std::move(regionSwitch);
		m_colormapSwitch = std::move(colormapSwitch);
	}

	std::shared_ptr<const DataFilter<float, 2>> source() const
	{
		return m_input->predecessor();
	}

	void setSource(std::shared_ptr<const DataFilter<float, 2>> source)
	{
		m_input->setPredecessor(source);
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

	std::shared_ptr<const fc::filter::Buffer<ndimdata::DataStatistic>> statistic() const
	{
		return m_statisticBuffer;
	}
	std::shared_ptr<const fc::filter::Buffer<QImage>> pixmap() const
	{
		return m_pixmapBuffer;
	}
};

} // namespace chains
} // namespace fc

#endif // FILTER_CHAINS_PIXMAPOUTPUT_H
