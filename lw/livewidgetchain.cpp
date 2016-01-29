#include "livewidgetchain.h"

#include "fc/filter.h"
#include "fc/validation/validator.h"
#include "fc/validation/watcher.h"

#include "fc/chains/darkimageopenbeam.h"
#include "fc/chains/sourceselect.h"
#include "fc/chains/dataprocess.h"
#include "fc/chains/pixmapoutput.h"
#include "fc/chains/profileplot.h"
#include "fc/chains/zplot.h"

#include "fc/buffer.h"
#include "ndimfilter/fits.h"

#include "safecast.h"

using jfh::assert_result;

namespace lw
{

struct LiveWidgetChain::LiveWidgetChainPrivate {
	fc::validation::Validator m_validator;

	fc::chains::DarkImageAndOpenBeamChain<float, 2> m_darkImageAndOpenBeam;
	std::shared_ptr<fc::fits::Loader<float, 2>> m_fileLoader;
	fc::chains::SourceSelectChain<float, 2> m_sourceSelect;
	fc::chains::DataProcessChain<float, 2> m_dataProcess;
	fc::chains::ImageOutputChain<float> m_imageOutput;
	fc::chains::ProfilePlotChain m_profileChain;
	fc::chains::ZPlotChain m_zPlotChain;

	fc::validation::Watcher m_processedDataWatcher;
	fc::validation::Watcher m_statisticWatcher;
	fc::validation::Watcher m_pixmapWatcher;
	fc::validation::Watcher m_profileWatcher;
	fc::validation::Watcher m_zStackWatcher;

	fc::chains::ImageOutputChain<float> m_testOutput;
	fc::validation::Watcher m_testWatcher;

	LiveWidgetChainPrivate()
		: m_darkImageAndOpenBeam()
		, m_fileLoader(fc::makeFitsLoader<float, 2>())
		, m_sourceSelect(m_darkImageAndOpenBeam.darkImageBuffer(), m_darkImageAndOpenBeam.openBeamBuffer(), m_fileLoader)
		, m_dataProcess(m_sourceSelect.sourceFilter())
		, m_imageOutput(m_dataProcess.processedBuffer())
		, m_profileChain(m_dataProcess.processedBuffer())
		, m_zPlotChain(m_darkImageAndOpenBeam.darkImageBuffer(), m_darkImageAndOpenBeam.openBeamBuffer())

		, m_processedDataWatcher(&m_validator, m_dataProcess.processedBuffer())
		, m_statisticWatcher(&m_validator, m_imageOutput.statistic())
		, m_pixmapWatcher(&m_validator, m_imageOutput.pixmap())
		, m_profileWatcher(&m_validator, m_profileChain.profileBuffer())
		, m_zStackWatcher(&m_validator, m_zPlotChain.stackBuffer())

		, m_testOutput(m_imageOutput.roiFilter())
		, m_testWatcher(&m_validator, m_testOutput.pixmap())

	{
		m_validator.add(m_darkImageAndOpenBeam.darkImageBuffer());
		m_validator.add(m_darkImageAndOpenBeam.openBeamBuffer());
		m_validator.add(m_imageOutput.pixmap());
		m_validator.add(m_testOutput.pixmap());
	}
};

LiveWidgetChain::LiveWidgetChain()
	: p(new LiveWidgetChainPrivate())
{
	assert_result(connect(&p->m_validator, SIGNAL(validationStarted()), this, SLOT(validator_validationStarted())));
	assert_result(connect(&p->m_validator, SIGNAL(validationStep()), this, SLOT(validator_validationStep())));
	assert_result(connect(&p->m_validator, SIGNAL(validationCompleted()), this, SLOT(validator_validationCompleted())));
	assert_result(connect(&p->m_validator, SIGNAL(invalidated()), this, SLOT(validator_invalidated())));
}

LiveWidgetChain::~LiveWidgetChain()
{
}

void LiveWidgetChain::setDarkImageFilenames(const QStringList &filenames)
{
	p->m_darkImageAndOpenBeam.setDarkImages(filenames);
}

void LiveWidgetChain::setOpenBeamFilenames(const QStringList &filenames)
{
	p->m_darkImageAndOpenBeam.setOpenBeam(filenames);
}

void LiveWidgetChain::setImageFilename(const QString &filename)
{
	p->m_fileLoader->setFilename(filename);
}

void LiveWidgetChain::setDataSource(LiveWidgetChain::DataSource source)
{
	p->m_sourceSelect.setSource(int(source));
}

void LiveWidgetChain::setNormalize(bool normalize)
{
	p->m_sourceSelect.setNormalize(normalize);
}

void LiveWidgetChain::setRegionOfInterest(ndim::range<2> region)
{
	p->m_imageOutput.setRegionOfInterest(region);
}

void LiveWidgetChain::resetRegionOfInterest()
{
	p->m_imageOutput.resetRegionOfInterest();
}

void LiveWidgetChain::setInvert(bool invert)
{
	p->m_imageOutput.setInvert(invert);
}

void LiveWidgetChain::setUseLog(bool log)
{
	p->m_imageOutput.setLog(log);
}

void LiveWidgetChain::setUseColor(bool color)
{
	p->m_imageOutput.setColor(color);
}

void LiveWidgetChain::setColorRange(std::pair<double, double> bounds)
{
	p->m_imageOutput.setColormapRange(bounds);
}

void LiveWidgetChain::setAutoColorRange(bool useFullRange)
{
	p->m_imageOutput.setAutoColormapRange(useFullRange);
}

bool LiveWidgetChain::isProcessedDataValid() const
{
	return p->m_dataProcess.processedBuffer()->isValid();
}

ndim::pointer<const float, 2> LiveWidgetChain::processedData() const
{
	return p->m_dataProcess.processedBuffer()->data();
}

void LiveWidgetChain::setProcessedDataCallback(std::function<void(ndim::pointer<const float, 2>)> callback)
{
	auto processedDataBuffer = p->m_dataProcess.processedBuffer();
	auto updater = [processedDataBuffer, callback]() { callback(processedDataBuffer->data()); };
	p->m_processedDataWatcher.setUpdater(updater);
}

bool LiveWidgetChain::isStatisticValid() const
{
	return p->m_imageOutput.statistic()->isValid();
}

const ndimdata::DataStatistic &LiveWidgetChain::statistic() const
{
	return p->m_imageOutput.statistic()->data().first();
}

void LiveWidgetChain::setStatisticCallback(std::function<void(const ndimdata::DataStatistic &)> callback)
{
	auto statisticBuffer = p->m_imageOutput.statistic();
	auto updater = [statisticBuffer, callback]() { callback(statisticBuffer->data().first()); };
	p->m_statisticWatcher.setUpdater(std::move(updater));
}

bool LiveWidgetChain::isPixmapValid() const
{
	return p->m_imageOutput.pixmap()->isValid();
}

const QPixmap &LiveWidgetChain::pixmap() const
{
	return p->m_imageOutput.pixmap()->data().first();
}

void LiveWidgetChain::setPixmapCallback(std::function<void(const QPixmap &)> callback)
{
	auto pixmapBuffer = p->m_imageOutput.pixmap();
	auto updater = [pixmapBuffer, callback]() { callback(pixmapBuffer->data().first()); };
	p->m_pixmapWatcher.setUpdater(std::move(updater));
}

void LiveWidgetChain::setProfileLine(QLineF line)
{
	p->m_profileChain.setLine(line);
}

bool LiveWidgetChain::isProfileValid() const
{
	return p->m_profileChain.profileBuffer()->isValid();
}

ndim::pointer<const float, 1> LiveWidgetChain::profile() const
{
	return p->m_profileChain.profileBuffer()->data();
}

void LiveWidgetChain::setProfileCallback(std::function<void(ndim::pointer<const float, 1>)> callback)
{
	auto profileBuffer = p->m_profileChain.profileBuffer();
	auto updater = [profileBuffer, callback]() { callback(profileBuffer->data()); };
	p->m_profileWatcher.setUpdater(updater);
}

void LiveWidgetChain::enableProfile(bool enable)
{
	if (enable)
		p->m_validator.add(p->m_profileChain.profileBuffer());
	else
		p->m_validator.remove(p->m_profileChain.profileBuffer().get());
}

void LiveWidgetChain::disableProfile()
{
	this->enableProfile(false);
}

void LiveWidgetChain::setZStackRegion(ndim::range<2> region)
{
	p->m_zPlotChain.setRange(region);
}

void LiveWidgetChain::setZStackFiles(const QStringList &filenames)
{
	p->m_zPlotChain.setFilenames(filenames);
}

bool LiveWidgetChain::isZStackValid() const
{
	return p->m_zPlotChain.stackBuffer()->isValid();
}

ndim::pointer<const float, 1> LiveWidgetChain::zStack() const
{
	return p->m_zPlotChain.stackBuffer()->data();
}

void LiveWidgetChain::setZStackCallback(std::function<void(ndim::pointer<const float, 1>)> callback)
{
	auto zStackBuffer = p->m_zPlotChain.stackBuffer();
	auto updater = [zStackBuffer, callback]() { callback(zStackBuffer->data()); };
	p->m_zStackWatcher.setUpdater(updater);
}

void LiveWidgetChain::enableZStack(bool enable)
{
	if (enable)
		p->m_validator.add(p->m_zPlotChain.stackBuffer());
	else
		p->m_validator.remove(p->m_zPlotChain.stackBuffer().get());
}

void LiveWidgetChain::disableZStack()
{
	this->enableZStack(false);
}

bool LiveWidgetChain::isTestPixmapValid() const
{
	return p->m_testOutput.pixmap()->isValid();
}

const QPixmap &LiveWidgetChain::testPixmap() const
{
	return p->m_testOutput.pixmap()->data().first();
}

void LiveWidgetChain::setTestPixmapCallback(std::function<void(const QPixmap &)> callback)
{
	auto pixmapBuffer = p->m_testOutput.pixmap();
	auto updater = [pixmapBuffer, callback]() { callback(pixmapBuffer->data().first()); };
	p->m_testWatcher.setUpdater(updater);
}

void LiveWidgetChain::start()
{
	p->m_validator.start();
}

void LiveWidgetChain::getStatus(size_t &progress, size_t &duration, size_t &step, size_t &stepCount, QString &description) const
{
	p->m_validator.state(progress, duration, step, stepCount, description);
}

QStringList LiveWidgetChain::getDescriptions() const
{
	return p->m_validator.descriptions();
}

void LiveWidgetChain::validator_validationStarted()
{
	emit validationStarted();
}

void LiveWidgetChain::validator_validationStep()
{
	emit validationStep();
}

void LiveWidgetChain::validator_validationCompleted()
{
	emit validationDone();
}

void LiveWidgetChain::validator_invalidated()
{
	emit invalidated();
}

} // namespace lw
