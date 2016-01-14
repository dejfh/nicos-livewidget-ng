
#include <type_traits>
#include <QVector>

#include "filter/fs/perelement.h"
#include "filter/fs/switch.h"
#include "filter/fs/buffer.h"
#include "filter/fs/accumulate.h"
#include "filter/fs/extend.h"
#include "filter/fs/fits.h"
#include "filter/fs/mapmerge.h"
#include "filter/fs/median.h"
#include "filter/fs/analyzer.h"
#include "filter/fs/pile.h"
#include "filter/fs/profile.h"
#include "filter/fs/subrange.h"
#include "filter/fs/pixmap.h"
#include "filter/fs/valuerange.h"
#include "filter/fs/input.h"

#include "helper/qt/vector.h"
#include "ndimdata/colormap.h"

void test3()
{
	auto fits = filter::fs::makeFitsLoader<float, 2>();
	auto input = filter::fs::makeInput<float, 2>();
	auto op = std::minus<float>();
	auto perElement = filter::fs::makePerElement("Hallo", op, fits, input);
	auto buffer = filter::fs::makeBuffer(perElement);
	auto sw = filter::fs::makeSwitch(buffer, perElement);
	auto i = ndim::makeIndices(1);
	auto acc = filter::fs::makeAccumulate(sw, std::plus<float>(), i, "Hallo Welt!");
	auto s = ndim::makeSizes(100);
	auto extend = filter::fs::makeExtend(acc, i, s);
	auto median = filter::fs::makeMedian("Blubb", extend, 1);
	auto v = hlp::makeQVector<std::shared_ptr<const filter::DataFilter<float, 1>>>(median);
	auto pile = filter::fs::makePile(v, 1);
	auto profile = filter::fs::makeProfile(pile, "Profile");
	auto subrange = filter::fs::makeSubrange(profile);
	hlp::unused(subrange);

	auto stat = filter::fs::makeAnalyzer(median, "lalala");
	auto valuerange = filter::fs::makeValueRange(stat);
	auto pixmap = filter::fs::makePixmap(pile, valuerange, ndimdata::ColorMapColor(0, 1), "Pixmap");
	hlp::unused(pixmap);

	auto map = filter::fs::makeFitsKeywordLoader();
	auto maps = //
		hlp::makeQVector<std::shared_ptr<const filter::DataFilter<QMap<QString, QString>>>>(map);
	auto merge = filter::fs::makeMapMerge(maps);

	hlp::unused(merge);
}
