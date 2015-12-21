#ifndef FILTER_CHAINS_PROFILEPLOT_H
#define FILTER_CHAINS_PROFILEPLOT_H

#include <memory>

#include "filter/filter.h"
#include "ndimfilter/profile.h"
#include "filter/buffer.h"
#include "ndimfilter/profile.h"

namespace filter
{
namespace chains
{

class ProfilePlotChain
{
	std::shared_ptr<filter::Profile> m_profile;
	std::shared_ptr<filter::Buffer<float, 1>> m_buffer;

public:
	ProfilePlotChain(std::shared_ptr<const filter::DataFilter<float, 2>> source)
	{
		m_profile = filter::makeProfile(source, "Generating Profile...");
		m_profile->setLine(QLineF(0, 0, 0, 0));
		m_buffer = filter::makeBuffer(m_profile);
	}

	void setLine(QLineF line)
	{
		m_profile->setLine(line);
	}

	std::shared_ptr<const filter::Buffer<float, 1>> profileBuffer() const
	{
		return m_buffer;
	}
};

} // namespace chains
} // namespace filter

#endif // FILTER_CHAINS_PROFILEPLOT_H
