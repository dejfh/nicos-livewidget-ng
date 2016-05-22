#ifndef FC_FILTER_FITS_H
#define FC_FILTER_FITS_H

#include <array>

#include <QDir>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "fits/fitsloader.h"

namespace fc
{
namespace filter
{

namespace fits
{

template <size_t Dimensionality>
class LoaderControl
{
public:
	virtual QString filename() const = 0;
	virtual void setFilename(QString filename) = 0;

	virtual bool isRangeSet() const = 0;
	virtual ndim::range<Dimensionality> range() const = 0;
	virtual void setRange(::ndim::range<Dimensionality> range) = 0;
	virtual void resetRange() = 0;
};

template <typename ElementType, size_t Dimensionality>
class Loader : public fc::FilterBase, public fc::DataFilter<ElementType, Dimensionality>, public LoaderControl<Dimensionality>
{
	struct Config {
		QString filename;
		bool isRangeSet;
		::ndim::range<Dimensionality> range;
	};

	hlp::Threadsafe<Config> m_config;

public:
	Loader(const QString &filename = QString())
	{
		m_config.unguardedMutable().filename = filename;
	}
	Loader(ndim::range<Dimensionality> range, const QString &filename = QString())
	{
		m_config.unguardedMutable().range = range;
		m_config.unguardedMutable().filename = filename;
	}

	QString filename() const override
	{
		return m_config.unguarded().filename;
	}
	void setFilename(QString filename) override
	{
		if (m_config.unguarded().filename == filename)
			return;
		this->invalidate();
		this->m_config.lock()->filename = std::move(filename);
	}

	bool isRangeSet() const override
	{
		return this->m_config.unguarded().isRangeSet;
	}
	ndim::range<Dimensionality> range() const override
	{
		return this->m_config.unguarded().range;
	}

	void resetRange() override
	{
		if (!this->m_config.unguarded().isRangeSet)
			return;
		this->invalidate();
		this->m_config.lock()->isRangeSet = false;
	}
	void setRange(ndim::range<Dimensionality> range) override
	{
		if (this->m_config.unguarded().isRangeSet && this->m_config.unguarded().range == range)
			return;
		this->invalidate();
		auto guard = this->m_config.lock();
		guard->range = range;
		guard->isRangeSet = true;
	}

	// DataFilter interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress) const override
	{
		Config config = m_config.get();
		progress.throwIfCancelled();

		ndim::sizes<Dimensionality> sizes = config.range.sizes;

		if (!config.isRangeSet)
			sizes = fitshelper::FitsHelper(config.filename).dimensions<Dimensionality>();

		progress.addStep(10 * sizes.size(), QString("Loading file: %1").arg(QDir(config.filename).dirName()));
		return sizes;
	}

	virtual ndim::Container<ElementType, Dimensionality> getData(
		ValidationProgress &progress, ndim::Container<ElementType, Dimensionality> *recycle) const override
	{
		Config config = m_config.get();
		progress.throwIfCancelled();

		fitshelper::FitsHelper fits(config.filename);
		ndim::sizes<Dimensionality> sizes = config.isRangeSet ? config.range.sizes : fits.dimensions<Dimensionality>();

		ndim::Container<ElementType, Dimensionality> result = ndim::makeMutableContainer(sizes, recycle);
		ndim::pointer<ElementType, Dimensionality> dataPointer = result.mutableData();

		if (config.isRangeSet)
			fits.read(dataPointer, config.range);
		else
			fits.read(dataPointer);

		progress.advanceProgress(10 * dataPointer.size());
		progress.advanceStep();

		return result;
	}
};

class KeywordLoaderOperation : public DataFilterHandlerBase<>
{
public:
	using ResultElementType = QMap<QString, QString>;
	static const size_t ResultDimensionality = 0;

	QString filename;

	KeywordLoaderOperation(const QString &filename = QString())
		: filename(filename)
	{
	}

	ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const
	{
		progress.addStep(10000, QString("Loading keys from file: %1").arg(QDir(filename).dirName()));
		return ndim::sizes<0>();
	}

	ndim::Container<ResultElementType, ResultDimensionality> getData(
		ValidationProgress &progress, ndim::Container<ResultElementType, ResultDimensionality> *recycle) const
	{
		fitshelper::FitsHelper fits(filename);

		ndim::Container<QMap<QString, QString>> result = ndim::makeMutableContainer(ndim::Sizes<0>(), recycle);

		result.mutableData().first() = fits.readUserKeys();

		progress.advanceProgress(10000);
		progress.advanceStep();

		return result;
	}
};

class KeywordLoader : public HandlerDataFilterBase<QMap<QString, QString>, 0, KeywordLoaderOperation>
{
public:
	KeywordLoader(const QString &filename = QString())
		: HandlerDataFilterBase<QMap<QString, QString>, 0, KeywordLoaderOperation>(KeywordLoaderOperation(filename))
	{
	}

	QString filename() const
	{
		return this->m_handler.unguarded().filename;
	}
	void setFilename(QString filename)
	{
		if (this->m_handler.unguarded().filename == filename)
			return;
		this->invalidate();
		this->m_handler.lock()->filename = std::move(filename);
	}
};

} // namespace fits

template <typename ElementType, size_t Dimensionality>
std::shared_ptr<fits::Loader<ElementType, Dimensionality>> makeFitsLoader(const QString &filename = QString())
{
	return std::make_shared<fits::Loader<ElementType, Dimensionality>>(filename);
}

inline std::shared_ptr<fits::KeywordLoader> makeFitsKeywordLoader(const QString &filename = QString())
{
	return std::make_shared<fits::KeywordLoader>(filename);
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_FITS_H
