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

void test2()
{
	auto fits = fc::filter::makeFitsLoader<float, 2>();
	auto input = fc::filter::makeInput<float, 2>();
	auto op = std::minus<float>();
	auto perElement = fc::filter::makePerElement("Hallo", op, fits, input);
	auto buffer = fc::filter::makeBuffer(perElement);
	auto sw = fc::filter::makeSwitch(buffer, perElement);
	auto i = ndim::makeIndices(1);
	auto acc = fc::filter::makeAccumulate(sw, std::plus<float>(), i, "Hallo Welt!");
	auto s = ndim::makeSizes(100);
	auto extend = fc::filter::makeExtend(acc, i, s);
	auto median = fc::filter::makeMedian("Blubb", extend, 1);
	auto v = hlp::makeQVector<std::shared_ptr<const fc::DataFilter<float, 1>>>(median);
	auto pile = fc::filter::makePile(v, 1);
	auto profile = fc::filter::makeProfile(pile, "Profile");
	auto subrange = fc::filter::makeSubrange(profile);
	hlp::unused(subrange);

	auto stat = fc::filter::makeAnalyzer(median, "lalala");
	auto valuerange = fc::filter::makeValueRange(stat);
	auto pixmap = fc::filter::makePixmap(pile, valuerange, ndimdata::ColorMapColor(0, 1), "Pixmap");
	hlp::unused(pixmap);

	auto map = fc::filter::makeFitsKeywordLoader();
	auto maps = //
		hlp::makeQVector<std::shared_ptr<const fc::DataFilter<QMap<QString, QString>>>>(map);
	auto merge = fc::filter::makeMapMerge(maps);

	hlp::unused(merge);
}
