#ifndef FILTER_CHAINS_PROFILEPLOT_H
#define FILTER_CHAINS_PROFILEPLOT_H

#include <memory>

#include "fc/filter.h"
#include "ndimfilter/profile.h"
#include "fc/buffer.h"
#include "ndimfilter/profile.h"

namespace fc
{
namespace chains
{

class ProfilePlotChain
{
	std::shared_ptr<fc::Profile> m_profile;
	std::shared_ptr<fc::Buffer<float, 1>> m_buffer;

public:
	ProfilePlotChain(std::shared_ptr<const fc::DataFilter<float, 2>> source)
	{
		m_profile = fc::makeProfile(source, "Generating Profile...");
		m_profile->setLine(QLineF(0, 0, 0, 0));
		m_buffer = fc::makeBuffer(m_profile);
	}

	void setLine(QLineF line)
	{
		m_profile->setLine(line);
	}

	std::shared_ptr<const fc::Buffer<float, 1>> profileBuffer() const
	{
		return m_buffer;
	}
};

} // namespace chains
} // namespace fc

#endif // FILTER_CHAINS_PROFILEPLOT_H
