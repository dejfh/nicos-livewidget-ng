#ifndef NDIMFILTER_FITS_H
#define NDIMFILTER_FITS_H

#include <array>
#include <utility>
#include <valarray>
#include <algorithm>
#include <atomic>

#include <QString>
#include <QDir>
#include <QMap>

#include "ndimfilter/filter.h"

#include "ndim/iterator.h"
#include "ndim/range.h"

#include "fits/fitsloader.h"

#include "helper/threadsafe.h"

namespace filter
{

namespace fits
{

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

template <size_t Dimensionality>
class LoaderControl
{
public:
	virtual QString filename() const = 0;
	virtual void setFilename(QString filename) = 0;

	virtual bool isRangeSet() const = 0;
	virtual ndim::range<Dimensionality> range() const = 0;
	virtual void setRange(::ndim::range<Dimensionality> range) = 0;
	virtual void setFullRange() = 0;
};

template <typename _ElementType, size_t _Dimensionality>
class Loader : public FilterBase,
			   public virtual NoConstDataFilter<_ElementType, _Dimensionality>,
			   public virtual NoConstDataFilter<QMap<QString, QString>>,
			   public virtual LoaderControl<_Dimensionality>
{
public:
	using ElementType = _ElementType;
	static const size_t Dimensionality = _Dimensionality;

private:
	hlp::Threadsafe<QString> m_filename;
	bool m_isRangeSet;
	::ndim::range<Dimensionality> m_range;

public:
	Loader(QString filename = QString())
		: m_filename(filename)
		, m_isRangeSet(false)
	{
	}

	virtual QString filename() const override
	{
		return m_filename.get();
	}
	virtual void setFilename(QString filename) override
	{
		if (m_filename.get() == filename)
			return;
		this->invalidate();
		m_filename = std::move(filename);
	}

	virtual bool isRangeSet() const override
	{
		return m_isRangeSet;
	}
	virtual ndim::range<Dimensionality> range() const override
	{
		return m_range;
	}
	virtual void setRange(ndim::range<Dimensionality> range) override
	{
		if (m_isRangeSet && m_range == range)
			return;
		this->invalidate();
		m_isRangeSet = true;
		m_range = range;
	}
	virtual void setFullRange() override
	{
		if (!m_isRangeSet)
			return;
		this->invalidate();
		m_isRangeSet = false;
	}

	std::shared_ptr<const DataFilter<ElementType, Dimensionality>> contentFilter() const
	{
		return std::shared_ptr<const DataFilter<ElementType, Dimensionality>>(this->shared_from_this(), this);
	}
	std::shared_ptr<const DataFilter<QMap<QString, QString>>> keyFilter() const
	{
		return std::shared_ptr<const DataFilter<QMap<QString, QString>>>(this->shared_from_this(), this);
	}

	// DataFilter<_T, _D> interface
public:
	virtual ndim::sizes<Dimensionality> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		QString filename = m_filename.get();
		bool isRangeSet = m_isRangeSet;
		ndim::sizes<Dimensionality> sizes = m_range.sizes;
		progress.throwIfCancelled();

		if (!isRangeSet)
			sizes = fitshelper::FitsHelper(filename).dimensions<Dimensionality>();

		progress.addStep(10 * sizes.size(), QString("Loading file: %1").arg(QDir(filename).dirName()));
		return sizes;
	}
	virtual void getData(ValidationProgress &progress, Container<ElementType, Dimensionality> &result,
		OverloadDummy<DataFilter<ElementType, Dimensionality>>) const override
	{
		QString filename = m_filename.get();
		bool isRangeSet = m_isRangeSet;
		auto range = m_range;
		progress.throwIfCancelled();

		fitshelper::FitsHelper fits(filename);
		ndim::sizes<Dimensionality> sizes = isRangeSet ? range.sizes : fits.dimensions<Dimensionality>();

		result.resize(sizes);
		ndim::pointer<ElementType, Dimensionality> dataPointer = result.pointer();

		if (isRangeSet)
			fits.read(dataPointer, range);
		else
			fits.read(dataPointer);

		progress.advanceProgress(10 * dataPointer.size());
		progress.advanceStep();
	}

	// DataFilter<QMap<QString, QString>> interface
public:
	virtual ndim::sizes<0> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<QMap<QString, QString>>>) const override
	{
		QString filename = m_filename.get();
		progress.throwIfCancelled();

		progress.addStep(10000, QString("Loading keys from file: %1").arg(QDir(filename).dirName()));
		return ndim::sizes<0>();
	}
	virtual void getData(
		ValidationProgress &progress, Container<QMap<QString, QString>> &result, OverloadDummy<DataFilter<QMap<QString, QString>>>) const override
	{
		QString filename = m_filename.get();
		progress.throwIfCancelled();

		fitshelper::FitsHelper fits(filename);
		result.resize();
		*result.pointer().data = fits.readUserKeys();

		progress.advanceProgress(10000);
		progress.advanceStep();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

} // namespace fits

template <typename _ElementType, size_t _Dimensions>
std::shared_ptr<fits::Loader<_ElementType, _Dimensions>> makeFitsLoader(const QString &filename = QString())
{
	return std::make_shared<fits::Loader<_ElementType, _Dimensions>>(filename);
}

} // namespace ndimfilter

#endif // NDIMFILTER_FITS_H
