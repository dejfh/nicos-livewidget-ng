#ifndef FC_FILTER_PIXMAP_H
#define FC_FILTER_PIXMAP_H

#include <QPixmap>
#include <QImage>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

#include "fc/gethelper.h"

#include "helper/helper.h"

#include "helper/helper.h"

namespace fc
{
namespace filter
{

template <typename _ElementType, typename _ColorMapType>
class PixmapHandler : public DataFilterHandlerBase<const DataFilter<_ElementType, 2>, const DataFilter<std::pair<double, double>>>
{
public:
	using ElementType = _ElementType;
	using ColorMapType = _ColorMapType;

	ColorMapType colormap;
	QString description;

	PixmapHandler(ColorMapType colormap, const QString &description)
		: colormap(colormap)
		, description(description)
	{
	}

	ndim::sizes<0> prepare(PreparationProgress &progress) const
	{
		auto sizes = hlp::throwIfNull(std::get<0>(this->predecessors))->prepare(progress);
		hlp::throwIfNull(std::get<1>(this->predecessors))->prepare(progress);

		progress.addStep(sizes.size(), description);

		return ndim::makeSizes();
	}
	ndim::Container<QImage> getData(ValidationProgress &progress, ndim::Container<QImage> *recycle) const
	{
		std::pair<double, double> range;
		auto data = std::get<0>(this->predecessors)->getData(progress);
		fc::getData(progress, std::get<1>(this->predecessors), range);

		auto dataPointer = data.constData();

		QImage image(int(dataPointer.width()), int(dataPointer.height()), QImage::Format_RGB32);
		ndim::pointer<QRgb, 2> imagePointer(hlp::cast_over_void<QRgb *>(image.bits()), dataPointer.sizes);
		imagePointer.byte_strides[1] = hlp::byte_offset_t(image.bytesPerLine());
		dataPointer.mirror(1);

#pragma omp parallel
		{
			ndim::transform_omp(imagePointer, _ColorMapType(range.first, range.second),
				dataPointer); // TODO: Currently does not use m_colormap. Introduce m_colormap.setRange(double, double)?
		}

		auto result = ndim::makeMutableContainer(recycle);

		result.mutableData().first() = std::move(image);

		progress.advanceProgress(dataPointer.size());
		progress.advanceStep();

		return result;
	}
};

template <typename _ElementType, typename _ColorMapType>
class Pixmap : public HandlerDataFilterWithDescriptionBase<QImage, 0, PixmapHandler<_ElementType, _ColorMapType>>
{
public:
	using ElementType = _ElementType;
	using ColorMapType = _ColorMapType;

	Pixmap(ColorMapType colormap, const QString &description)
		: HandlerDataFilterWithDescriptionBase<QImage, 0, PixmapHandler<_ElementType, _ColorMapType>>(colormap, description)
	{
	}

	ColorMapType colormap() const
	{
		return this->m_handler.unguarded().colormap;
	}
	void setColorMap(ColorMapType colormap)
	{
		this->invalidate();
		this->m_handler.lock()->colormap = colormap;
	}

	std::shared_ptr<const DataFilter<ElementType, 2>> dataFilter() const
	{
		return this->template predecessor<0>();
	}
	void setDataFilter(std::shared_ptr<const DataFilter<ElementType, 2>> dataFilter)
	{
		this->template setPredecessor<0>(std::move(dataFilter));
	}

	std::shared_ptr<const DataFilter<std::pair<double, double>>> rangeFilter() const
	{
		return this->template predecessor<1>();
	}
	void setRangeFilter(std::shared_ptr<const DataFilter<std::pair<double, double>>> rangeFilter)
	{
		this->template setPredecessor<1>(std::move(rangeFilter));
	}
};

template <typename DataFilterType, typename ColorMapType>
std::shared_ptr<Pixmap<ElementTypeOf_t<DataFilterType>, ColorMapType>> makePixmap(std::shared_ptr<DataFilterType> dataFilter,
	std::shared_ptr<const DataFilter<std::pair<double, double>>> rangeFilter, ColorMapType colormap, const QString &description)
{
	auto filter = std::make_shared<Pixmap<ElementTypeOf_t<DataFilterType>, ColorMapType>>(colormap, description);
	filter->setDataFilter(dataFilter);
	filter->setRangeFilter(rangeFilter);
	return filter;
}

} // namespace filter
} // namespace fc
#endif // FC_FILTER_PIXMAP_H
