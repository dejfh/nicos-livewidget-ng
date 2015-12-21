#ifndef NDIMFILTER_PROFILE_H
#define NDIMFILTER_PROFILE_H

#include "ndim/pointer.h"
#include "ndim/buffer.h"
#include "ndim/iterator.h"

#include "filter/filter.h"
#include "filter/filterbase.h"
#include "ndimfilter/filter.h"

#include "filter/gethelper.h"

#include <QList>
#include <QString>
#include <QPointF>
#include <QLineF>
#include <QPoint>

#include <array>

#include <cmath>

#include <algorithm>

#include <helper/helper.h>

namespace filter
{

class Profile : public SinglePredecessorFilterBase<DataFilter<float, 2>>, public NoConstDataFilter<float, 1>
{
	QLineF m_line;
	int m_lineWidth;
	int m_binWidth;

	const QString m_description;

public:
	Profile(QLineF line, int lineWidth, int binWidth, const QString &description)
		: m_line(line)
		, m_lineWidth(lineWidth)
		, m_binWidth(binWidth)
		, m_description(description)
	{
	}

	QLineF line() const
	{
		return m_line;
	}
	void setLine(QLineF line)
	{
		this->setAndInvalidate(m_line, line);
	}

	int lineWidth() const
	{
		return m_lineWidth;
	}
	void setLineWidth(int lineWidth)
	{
		this->setAndInvalidate(m_lineWidth, lineWidth);
	}

	int binWidth() const
	{
		return m_binWidth;
	}
	void setBinWidth(int binWidth)
	{
		this->setAndInvalidate(m_binWidth, binWidth);
	}

	// DataFilter interface
public:
	virtual ndim::sizes<1> prepare(PreparationProgress &progress, OverloadDummy<DataFilter<float, 1>>) const override
	{
		auto predecessor = this->predecessor();
		QLineF line = m_line;
		QPointF start = line.p1() - QPointF(.5, .5);
		QPointF end = line.p2() - QPointF(.5, .5);
		int lineWidth = m_lineWidth;
		progress.throwIfCancelled();

		predecessor->prepareConst(progress);

		QPointF diff = end - start;
		double len = std::sqrt(diff.x() * diff.x() + diff.y() * diff.y());

		size_t count = (size_t)(len + 0.5);

		progress.addStep(size_t(line.length() * lineWidth), m_description);
		return ndim::sizes<1>{count};
	}
	virtual void getData(ValidationProgress &progress, Container<float, 1> &result, OverloadDummy<DataFilter<float, 1>>) const override
	{
		auto predecessor = this->predecessor();
		QLineF line = m_line;
		QPointF start = line.p1() - QPointF(.5, .5);
		QPointF end = line.p2() - QPointF(.5, .5);
		int lineWidth = m_lineWidth;
		int binWidth = m_binWidth;
		progress.throwIfCancelled();

		Container<const float, 2> container = filter::getConstData(progress, predecessor);
		ndim::pointer<const float, 2> pointer = container.pointer();

		{
			using namespace std;

			QPointF diff = end - start;
			double len = sqrt(diff.x() * diff.x() + diff.y() * diff.y());
			double angle = atan2(diff.y(), diff.x());

			size_t count = (int)(len + 0.5);
			result.resize(ndim::sizes<1>{count});

			lineWidth = max(int(1), lineWidth);

			QPointF dirL(cos(angle), sin(angle));
			QPointF dirO(dirL.y(), -dirL.x());

			QPointF posSide = start - (lineWidth - 1) / 2. * dirO;

			binWidth = max(1, binWidth);

			ndim::iterator<float, 1> it(result.pointer());

			for (size_t i = 0; i < count; ++i, posSide += dirL, ++it) {
				float value = 0;
				QPointF pos = posSide;
				for (int y = 0; y < lineWidth; y++, pos += dirO) {
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

					value += float(w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11);
				}
				*it = value;
			}
		}

		progress.advanceProgress(size_t(line.length() * lineWidth));
		progress.advanceStep();
	}
};

inline std::shared_ptr<Profile> makeProfile(std::shared_ptr<const DataFilter<float, 2>> predecessor, QString description)
{
	auto filter = std::make_shared<Profile>(QLineF(), 0, 0, std::move(description));
	filter->setPredecessor(std::move(predecessor));
	return filter;
}

} // namespace ndimfilter

#endif // NDIMFILTER_PROFILE_H
