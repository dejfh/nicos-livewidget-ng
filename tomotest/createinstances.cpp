#include <type_traits>
#include <QVector>

#include "fc/filter/perelement.h"
#include "fc/filter/switch.h"
#include "fc/filter/buffer.h"
#include "fc/filter/accumulate.h"
#include "fc/filter/extend.h"
#include "fc/filter/fits.h"
#include "fc/filter/mapmerge.h"
#include "fc/filter/median.h"
#include "fc/filter/analyzer.h"
#include "fc/filter/pile.h"
#include "fc/filter/profile.h"
#include "fc/filter/subrange.h"
#include "fc/filter/pixmap.h"
#include "fc/filter/valuerange.h"
#include "fc/filter/input.h"

#include "helper/qt/vector.h"
#include "ndimdata/colormap.h"

void test1()
{
	std::shared_ptr<const fc::DataFilter<float, 2>> fits = fc::filter::makeFitsLoader<float, 2>();
	std::shared_ptr<const fc::DataFilter<float, 2>> input = fc::filter::makeInput<float, 2>();
	auto op = std::minus<float>();
	std::shared_ptr<const fc::DataFilter<float, 2>> perElement = fc::filter::makePerElement("Hallo", op, fits, input);
	std::shared_ptr<const fc::filter::Buffer<float, 2>> buffer = fc::filter::makeBuffer(perElement);
	std::shared_ptr<const fc::DataFilter<float, 2>> sw = fc::filter::makeSwitch(buffer, perElement);
	auto i = ndim::makeIndices(1);
	std::shared_ptr<const fc::DataFilter<float, 1>> acc = fc::filter::makeAccumulate(sw, std::plus<float>(), i, "Hallo Welt!");
	auto s = ndim::makeSizes(100);
	std::shared_ptr<const fc::DataFilter<float, 2>> extend = fc::filter::makeExtend(acc, i, s);
	std::shared_ptr<const fc::DataFilter<float, 1>> median = fc::filter::makeMedian("Blubb", extend, 1);
	QVector<std::shared_ptr<const fc::DataFilter<float, 1>>> v = hlp::makeQVector<std::shared_ptr<const fc::DataFilter<float, 1>>>(median);
	std::shared_ptr<const fc::DataFilter<float, 2>> pile = fc::filter::makePile(v, 1);
	std::shared_ptr<const fc::DataFilter<float, 1>> profile = fc::filter::makeProfile(pile, "Profile");
	std::shared_ptr<const fc::DataFilter<float, 1>> subrange = fc::filter::makeSubrange(profile);
	hlp::unused(subrange);

	std::shared_ptr<const fc::DataFilter<ndimdata::DataStatistic>> stat = fc::filter::makeAnalyzer(median, "lalala");
	std::shared_ptr<const fc::DataFilter<std::pair<double, double>>> valuerange = fc::filter::makeValueRange(stat);
	std::shared_ptr<const fc::DataFilter<QImage>> pixmap = fc::filter::makePixmap(pile, valuerange, ndimdata::ColorMapColor(0, 1), "Pixmap");
	hlp::unused(pixmap);

	std::shared_ptr<const fc::DataFilter<QMap<QString, QString>>> map = fc::filter::makeFitsKeywordLoader();
	QVector<std::shared_ptr<const fc::DataFilter<QMap<QString, QString>>>> maps =
		hlp::makeQVector<std::shared_ptr<const fc::DataFilter<QMap<QString, QString>>>>(map);
	std::shared_ptr<const fc::DataFilter<QMap<QString, QString>>> merge = fc::filter::makeMapMerge(maps);

	hlp::unused(merge);
}
