#ifndef filter_PIXMAPBUILDER_H
#define filter_PIXMAPBUILDER_H

#include "filter/filter.h"

#include <utility>

#include <QPixmap>
#include <QImage>

#include "helper/helper.h"
#include "filter/gethelper.h"

#include "filter/filter.h"
#include "filter/filterbase.h"

#include "ndimdata/statistic.h"

#include "ndim/algorithm_omp.h"

#include "safecast.h"

using jfh::assert_cast;
using jfh::cast_over_void;

namespace filter
{

enum Colormaps { ColormapGrayscale, ColormapColor, ColormapCyclic };

#ifdef Q_CC_MSVC
#pragma warning(push)
// Disable inherits via dominance warning.
// Bug in Visual C++, since there is only one implementation there should be no warning.
#pragma warning(disable : 4250)
#endif

class PixmapRange : public SinglePredecessorFilterBase<DataFilter<ndimdata::DataStatistic>>,
					public virtual NoConstDataFilter<std::pair<double, double>>
{
	std::pair<double, double> m_range;
	bool m_isSet;
	bool m_useFullRange;
	bool m_invert;

public:
	PixmapRange()
		: m_isSet(false)
		, m_useFullRange(false)
		, m_invert(false)
	{
	}

	std::pair<double, double> range() const
	{
		return m_range;
	}
	void setRange(std::pair<double, double> range)
	{
		if (m_isSet && m_range == range)
			return;
		this->invalidate();
		m_isSet = true;
		m_range = range;
	}

	void setAuto(bool useFullRange)
	{
		if (!m_isSet && m_useFullRange == useFullRange)
			return;
		this->invalidate();
		m_isSet = false;
		m_useFullRange = useFullRange;
	}

	bool invert() const
	{
		return m_invert;
	}
	void setInvert(bool invert)
	{
		if (m_invert == invert)
			return;
		this->invalidate();
		m_invert = invert;
	}

	// Invalidatable interface
public:
	virtual void predecessorInvalidated(const Predecessor *) override
	{
		if (!m_isSet)
			this->invalidate();
	}

	// DataFilter interface
public:
	virtual ndim::sizes<0> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<std::pair<double, double>>>) const override
	{
		bool isSet = m_isSet;
		auto predecessor = this->predecessor();
		progress.throwIfCancelled();
		hlp::notNull(predecessor);

		if (!isSet)
			predecessor->prepareConst(progress);

		return ndim::sizes<0>();
	}
	virtual void getData(ValidationProgress &progress, Container<std::pair<double, double>> &result,
		OverloadDummy<DataFilter<std::pair<double, double>>>) const override
	{
		auto predecessor = this->predecessor();
		std::pair<double, double> range = m_range;
		bool isSet = m_isSet;
		bool useFullRange = m_useFullRange;
		bool invert = m_invert;
		progress.throwIfCancelled();

		auto &data = *hlp::notNull(result.reset(ndim::sizes<0>()).data);

		if (isSet)
			data = range;
		else {
			auto container = filter::getConstData(progress, predecessor);
			const ndimdata::DataStatistic &stat = *hlp::notNull(container.pointer().data);

			if (useFullRange)
				data = std::make_pair(stat.min, stat.max);
			else
				data = std::make_pair(stat.auto_low_bound, stat.auto_high_bound);
		}

		if (invert)
			std::swap(data.first, data.second);
	}
};

inline std::shared_ptr<PixmapRange> makePixmapRange(std::shared_ptr<const DataFilter<ndimdata::DataStatistic>> predecessor)
{
	auto filter = std::make_shared<PixmapRange>();
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

template <typename _ElementType, typename _ColorMapType>
class PixmapBuilder : public FilterBase, public NoConstDataFilter<QPixmap>
{
	using ElementType = _ElementType;
	using ColorMapType = _ColorMapType;

	using DataFilterType = const DataFilter<ElementType, 2>;
	using RangeFilterType = const DataFilter<std::pair<double, double>>;

	PredecessorStore<DataFilterType> m_dataFilter;
	PredecessorStore<RangeFilterType> m_rangeFilter;

	const ColorMapType m_colormap;
	const QString m_description;

public:
	PixmapBuilder(ColorMapType colormap, QString description)
		: m_colormap(std::move(colormap))
		, m_description(std::move(description))
	{
	}

	virtual ~PixmapBuilder()
	{
		m_dataFilter.clear(this);
		m_rangeFilter.clear(this);
	}

	std::shared_ptr<DataFilterType> dataFilter() const
	{
		return m_dataFilter.get();
	}
	std::shared_ptr<DataFilterType> setDataFilter(std::shared_ptr<DataFilterType> dataFilter)
	{
		return m_dataFilter.reset(std::move(dataFilter), this->shared_from_this());
	}
	std::shared_ptr<RangeFilterType> rangeFilter() const
	{
		return m_rangeFilter.get();
	}
	std::shared_ptr<RangeFilterType> setRangeFilter(std::shared_ptr<RangeFilterType> rangeFilter)
	{
		return m_rangeFilter.reset(std::move(rangeFilter), this->shared_from_this());
	}

	// DataFilter interface
public:
	virtual ndim::sizes<0> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<QPixmap>>) const override
	{
		auto rangeFilter = m_rangeFilter.get();
		auto dataFilter = m_dataFilter.get();
		progress.throwIfCancelled();
		hlp::notNull(rangeFilter);
		hlp::notNull(dataFilter);

		rangeFilter->prepareConst(progress);
		auto sizes = dataFilter->prepareConst(progress);

		progress.addStep(sizes.size(), m_description);
		return ndim::sizes<0>();
	}
	virtual void getData(ValidationProgress &progress, Container<QPixmap> &result,
		OverloadDummy<DataFilter<QPixmap>>) const override
	{
		auto rangeFilter = m_rangeFilter.get();
		auto dataFilter = m_dataFilter.get();
		progress.throwIfCancelled();
		hlp::notNull(rangeFilter);
		hlp::notNull(dataFilter);

		Container<const std::pair<double, double>> rangeContainer = filter::getConstData(progress, rangeFilter);
		const std::pair<double, double> &range = *rangeContainer.pointer().data;
		Container<const ElementType, 2> data = filter::getConstData(progress, dataFilter);
		auto dataPointer = data.pointer();

		QImage image(assert_cast<int>(dataPointer.width()), assert_cast<int>(dataPointer.height()), QImage::Format_RGB32);
		ndim::pointer<QRgb, 2> imagePointer(cast_over_void<QRgb *>(image.bits()), dataPointer.sizes);
		imagePointer.strides[1] = image.bytesPerLine() / sizeof(QRgb);
		dataPointer.mirror(1);

#pragma omp parallel
		{
			ndim::transform_omp(imagePointer, _ColorMapType(range.first, range.second),
				dataPointer); // TODO: Currently does not use m_colormap. Introduce m_colormap.setRange(double, double)?
		}

		result.resize();
		*result.pointer().data = QPixmap::fromImage(image);

		progress.advanceProgress(dataPointer.size());
		progress.advanceStep();
	}
};

#ifdef Q_CC_MSVC
#pragma warning(pop)
#endif

template <typename _ElementType, typename _ColorMapType>
std::shared_ptr<PixmapBuilder<_ElementType, _ColorMapType>> _makePixmapBuilder(std::shared_ptr<const DataFilter<_ElementType, 2>> dataFilter,
	std::shared_ptr<const DataFilter<std::pair<double, double>>> rangeFilter, _ColorMapType colormap, QString description)
{
	auto filter = std::make_shared<PixmapBuilder<_ElementType, _ColorMapType>>(colormap, description);
	filter->setDataFilter(std::move(dataFilter));
	filter->setRangeFilter(std::move(rangeFilter));
	return filter;
}

template <typename _Predecessor, typename _ColorMapType>
auto makePixmapBuilder(std::shared_ptr<_Predecessor> dataFilter, std::shared_ptr<const DataFilter<std::pair<double, double>>> rangeFilter,
	_ColorMapType colormap, QString description)
	-> decltype(_makePixmapBuilder(asDataFilter(std::move(dataFilter)), rangeFilter, colormap, std::move(description)))
{
	return _makePixmapBuilder(asDataFilter(std::move(dataFilter)), rangeFilter, colormap, std::move(description));
}

} // namespace filter

#endif // filter_PIXMAPBUILDER_H
