#ifndef FILTER_CHAINS_FITSPILE_H
#define FILTER_CHAINS_FITSPILE_H

#include <memory>

#include "filter/filter.h"
#include "filter/fs/fits.h"
#include "filter/fs/pile.h"

namespace filter
{
namespace chains
{

template <typename _ElementType, size_t _Dimensionality>
class FitsPileChain
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	QVector<std::shared_ptr<filter::fs::fits::Loader<ElementType, Dimensionality>>> m_fits;
	std::shared_ptr<filter::fs::Pile<ElementType, Dimensionality>> m_pile;

	ndim::range<Dimensionality> m_range;
	bool m_isRangeSet;

public:
	FitsPileChain()
		: m_isRangeSet(false)
	{
		m_pile = filter::makePile<ElementType, Dimensionality>(0);
	}

	void setFilenames(const QStringList &filenames)
	{
		m_fits.clear();
		m_fits.reserve(filenames.size());

		auto fitsGenerator = [this](const QString &filename) {
			auto fits = filter::makeFitsLoader<ElementType, Dimensionality>(filename);
			if (this->m_isRangeSet)
				fits->setRange(m_range);
			return fits;
		};

		std::transform(filenames.cbegin(), filenames.cend(), std::back_inserter(m_fits), fitsGenerator);

		QVector<std::shared_ptr<const filter::DataFilter<ElementType, Dimensionality>>> filters;
		filters.reserve(m_fits.size());

		std::copy(m_fits.cbegin(), m_fits.cend(), std::back_inserter(filters));

		m_pile->setPredecessors(filters);
	}

	void resetRange()
	{
		m_isRangeSet = false;
		for (const auto &fits : m_fits)
			fits->resetRange();
	}
	void setRange(ndim::range<Dimensionality> range)
	{
		m_isRangeSet = true;
		m_range = range;
		for (const auto &fits : m_fits)
			fits->setRange(range);
	}

	bool isRangeSet() const
	{
		return m_isRangeSet;
	}
	ndim::range<Dimensionality> range() const
	{
		return m_range;
	}

	std::shared_ptr<const filter::DataFilter<ElementType, Dimensionality + 1>> pileBuffer()
	{
		return m_pile;
	}
};

} // namespace chains
} // namespace filter

#endif // FILTER_CHAINS_FITSPILE_H
