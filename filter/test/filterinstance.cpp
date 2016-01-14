
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
	std::shared_ptr<const filter::DataFilter<float, 2>> fits = filter::fs::makeFitsLoader<float, 2>();
	std::shared_ptr<const filter::DataFilter<float, 2>> input = filter::fs::makeInput<float, 2>();
	auto op = std::minus<float>();
	std::shared_ptr<const filter::DataFilter<float, 2>> perElement = filter::fs::makePerElement("Hallo", op, fits, input);
	std::shared_ptr<const filter::fs::Buffer<float, 2>> buffer = filter::fs::makeBuffer(perElement);
	std::shared_ptr<const filter::DataFilter<float, 2>> sw = filter::fs::makeSwitch(buffer, perElement);
	auto i = ndim::makeIndices(1);
	std::shared_ptr<const filter::DataFilter<float, 1>> acc = filter::fs::makeAccumulate(sw, std::plus<float>(), i, "Hallo Welt!");
	auto s = ndim::makeSizes(100);
	std::shared_ptr<const filter::DataFilter<float, 2>> extend = filter::fs::makeExtend(acc, i, s);
	std::shared_ptr<const filter::DataFilter<float, 1>> median = filter::fs::makeMedian("Blubb", extend, 1);
	QVector<std::shared_ptr<const filter::DataFilter<float, 1>>> v = hlp::makeQVector<std::shared_ptr<const filter::DataFilter<float, 1>>>(median);
	std::shared_ptr<const filter::DataFilter<float, 2>> pile = filter::fs::makePile(v, 1);
	std::shared_ptr<const filter::DataFilter<float, 1>> profile = filter::fs::makeProfile(pile, "Profile");
	std::shared_ptr<const filter::DataFilter<float, 1>> subrange = filter::fs::makeSubrange(profile);
	hlp::unused(subrange);

	std::shared_ptr<const filter::DataFilter<ndimdata::DataStatistic>> stat = filter::fs::makeAnalyzer(median, "lalala");
	std::shared_ptr<const filter::DataFilter<std::pair<double, double>>> valuerange = filter::fs::makeValueRange(stat);
	std::shared_ptr<const filter::DataFilter<QPixmap>> pixmap = filter::fs::makePixmap(pile, valuerange, ndimdata::ColorMapColor(0, 1), "Pixmap");
	hlp::unused(pixmap);

	std::shared_ptr<const filter::DataFilter<QMap<QString, QString>>> map = filter::fs::makeFitsKeywordLoader();
	QVector<std::shared_ptr<const filter::DataFilter<QMap<QString, QString>>>> maps =
		hlp::makeQVector<std::shared_ptr<const filter::DataFilter<QMap<QString, QString>>>>(map);
	std::shared_ptr<const filter::DataFilter<QMap<QString, QString>>> merge = filter::fs::makeMapMerge(maps);

	hlp::unused(merge);
}
