#ifndef FILTER_FS_FITS_H
#define FILTER_FS_FITS_H

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

template <typename _ElementType, size_t _Dimensionality>
class LoaderHandler : public DataFilterHandlerBase<>
{
public:
	using ResultElementType = _ElementType;
	static const size_t ResultDimensionality = _Dimensionality;

	QString filename;
	bool isRangeSet;
	::ndim::range<ResultDimensionality> range;

	LoaderHandler(const QString &filename = QString())
		: filename(filename)
		, isRangeSet(false)
	{
	}
	LoaderHandler(ndim::range<ResultDimensionality> range, const QString &filename = QString())
		: filename(filename)
		, isRangeSet(true)
		, range(range)
	{
	}

	ndim::sizes<ResultDimensionality> prepare(PreparationProgress &progress) const
	{
		progress.throwIfCancelled();

		ndim::sizes<ResultDimensionality> sizes = range.sizes;

		if (!isRangeSet)
			sizes = fitshelper::FitsHelper(filename).dimensions<ResultDimensionality>();

		progress.addStep(10 * sizes.size(), QString("Loading file: %1").arg(QDir(filename).dirName()));
		return sizes;
	}

	Container<ResultElementType, ResultDimensionality> getData(
		ValidationProgress &progress, Container<ResultElementType, ResultDimensionality> *recycle) const
	{
		progress.throwIfCancelled();

		fitshelper::FitsHelper fits(filename);
		ndim::sizes<ResultDimensionality> sizes = isRangeSet ? range.sizes : fits.dimensions<ResultDimensionality>();

		Container<ResultElementType, ResultDimensionality> result = fc::makeMutableContainer(sizes, recycle);
		ndim::pointer<ResultElementType, ResultDimensionality> dataPointer = result.mutableData();

		if (isRangeSet)
			fits.read(dataPointer, range);
		else
			fits.read(dataPointer);

		progress.advanceProgress(10 * dataPointer.size());
		progress.advanceStep();

		return result;
	}
};

template <typename ElementType, size_t Dimensionality>
class Loader : public HandlerDataFilterBase<ElementType, Dimensionality, LoaderHandler<ElementType, Dimensionality>>,
			   public LoaderControl<Dimensionality>
{
public:
	Loader(const QString &filename = QString())
		: HandlerDataFilterBase<ElementType, Dimensionality, LoaderHandler<ElementType, Dimensionality>>(
			  LoaderHandler<ElementType, Dimensionality>(filename))
	{
	}
	Loader(ndim::range<Dimensionality> range, const QString &filename = QString())
		: HandlerDataFilterBase<ElementType, Dimensionality, LoaderHandler<ElementType, Dimensionality>>(
			  LoaderHandler<ElementType, Dimensionality>(range, filename))
	{
	}

	QString filename() const override
	{
		return this->m_handler.unguarded().filename;
	}
	void setFilename(QString filename) override
	{
		if (this->m_handler.unguarded().filename == filename)
			return;
		this->invalidate();
		this->m_handler.lock()->filename = std::move(filename);
	}

	bool isRangeSet() const override
	{
		return this->m_handler.unguarded().isRangeSet;
	}
	ndim::range<Dimensionality> range() const override
	{
		return this->m_handler.unguarded().range;
	}

	void resetRange() override
	{
		if (!this->m_handler.unguarded().isRangeSet)
			return;
		this->invalidate();
		this->m_handler.lock()->isRangeSet = false;
	}
	void setRange(ndim::range<Dimensionality> range) override
	{
		if (this->m_handler.unguarded().isRangeSet && this->m_handler.unguarded().range == range)
			return;
		this->invalidate();
		auto guard = this->m_handler.lock();
		guard->range = range;
		guard->isRangeSet = true;
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

	Container<ResultElementType, ResultDimensionality> getData(
		ValidationProgress &progress, Container<ResultElementType, ResultDimensionality> *recycle) const
	{
		fitshelper::FitsHelper fits(filename);

		Container<QMap<QString, QString>> result = fc::makeMutableContainer(ndim::Sizes<0>(), recycle);

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

#endif // FILTER_FS_FITS_H
