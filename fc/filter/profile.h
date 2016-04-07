#ifndef FC_FILTER_PROFILE_H
#define FC_FILTER_PROFILE_H

#include <QLineF>

#include "fc/datafilter.h"
#include "fc/datafilterbase.h"

namespace fc
{
namespace filter
{

template <typename _ElementType>
class ProfileHandler : public DataFilterHandlerBase<const DataFilter<_ElementType, 2>>
{
public:
	using ElementType = _ElementType;

	QLineF line;
	size_t lineWidth;
	size_t binWidth;
	QString description;

	ProfileHandler(QLineF line, size_t lineWidth, size_t binWidth, const QString &description)
		: line(line)
		, lineWidth(lineWidth)
		, binWidth(binWidth)
		, description(description)
	{
	}

	ndim::sizes<1> prepare(PreparationProgress &progress) const
	{
		this->preparePredecessors(progress);
		size_t count = size_t(line.length() + 1.5);
		size_t lineWidth = std::max(size_t(1), this->lineWidth);
		progress.addStep(count * lineWidth, description);
		return ndim::makeSizes(count);
	}
	ndim::Container<ElementType, 1> getData(ValidationProgress &progress, ndim::Container<ElementType, 1> *recycle) const
	{
		QPointF start = line.p1() - QPointF(.5, .5);
		QPointF end = (line.p2() - QPointF(.5, .5));
		size_t lineWidth = std::max(size_t(1), this->lineWidth);

		// TODO: binWidth currentliy ignored
		//		size_t binWidth = std::max(size_t(1), this->binWidth);

		auto input = std::get<0>(this->getPredecessorsData(progress));
		ndim::pointer<const ElementType, 2> pointer = input.constData();

		{
			using namespace std;

			QPointF diff = end - start;
			double angle = atan2(diff.y(), diff.x());

			size_t count = size_t(line.length() + 1.5);

			QPointF dirL(cos(angle), sin(angle));
			QPointF dirO(dirL.y(), -dirL.x());

			QPointF posSide = start - (lineWidth - 1) / 2.0 * dirO;

			ndim::Container<float, 1> result = ndim::makeMutableContainer(ndim::Sizes<1>{count}, recycle);
			ndim::iterator<float, 1> it(result.mutableData());

			for (size_t i = 0; i < count; ++i, posSide += dirL, ++it) {
				float value = 0;
				QPointF pos = posSide;
				for (size_t y = 0; y < lineWidth; y++, pos += dirO) {
					QPoint iPos(floor(pos.x()), floor(pos.y()));
					QPointF subPos(pos - iPos);

					// area weighting
					double w00 = (1 - subPos.x()) * (1 - subPos.y());
					double w10 = (subPos.x()) * (1 - subPos.y());
					double w01 = (1 - subPos.x()) * (subPos.y());
					double w11 = (subPos.x()) * (subPos.y());

					double v00 = pointer.value(0, iPos.x(), iPos.y());
					double v10 = pointer.value(0, iPos.x() + 1, iPos.y());
					double v01 = pointer.value(0, iPos.x(), iPos.y() + 1);
					double v11 = pointer.value(0, iPos.x() + 1, iPos.y() + 1);

					value += ElementType(w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11);
				}
				*it = value;
			}

			progress.advanceProgress(size_t(line.length() * lineWidth));
			progress.advanceStep();

			return result;
		}
	}
};

template <typename _ElementType>
class Profile : public HandlerDataFilterWithDescriptionBase<_ElementType, 1, ProfileHandler<_ElementType>>
{
public:
	Profile(QLineF line, size_t lineWidth, size_t binWidth, const QString &description)
		: HandlerDataFilterWithDescriptionBase<_ElementType, 1, ProfileHandler<_ElementType>>(line, lineWidth, binWidth, description)
	{
	}

	QLineF line() const
	{
		return this->m_handler.unguarded().line;
	}
	void setLine(QLineF line)
	{
		if (this->m_handler.unguarded().line == line)
			return;
		this->invalidate();
		this->m_handler.lock()->line = line;
	}

	size_t lineWidth() const
	{
		return this->m_handler.unguarded().lineWidth;
	}
	void setLineWidth(size_t lineWidth)
	{
		if (this->m_handler.unguarded().lineWidth == lineWidth)
			return;
		this->invalidate();
		this->m_handler.lock()->lineWidth = lineWidth;
	}

	size_t binWidth() const
	{
		return this->m_handler.unguarded().binWidth;
	}
	void setBinWidth(size_t binWidth)
	{
		if (this->m_handler.unguarded().binWidth == binWidth)
			return;
		this->invalidate();
		this->m_handler.lock()->binWidth = binWidth;
	}
};

template <typename PredecessorType>
std::shared_ptr<Profile<ElementTypeOf_t<PredecessorType>>> makeProfile(std::shared_ptr<PredecessorType> predecessor, QString description)
{
	auto filter = std::make_shared<Profile<ElementTypeOf_t<PredecessorType>>>(QLineF(), 0, 0, std::move(description));
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace filter
} // namespace fc

#endif // FC_FILTER_PROFILE_H
