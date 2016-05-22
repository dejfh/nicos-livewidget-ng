//#include "filterchain.h"

//#include <utility> // std::make_pair

//#include <pyfc/numpy.h>

//#include <iostream>

//#include <QImage>

//#include "helper/helper.h"
//#include "helper/python/gilhelper.h"

//#include "numpyinput.h"

//#include "fc/filter/fits.h"
//#include "fc/filter/median.h"

//#include "fc/filter/buffer.h"
//#include "fc/filter/correction.h"

//using hlp::assert_true;

//FilterChain::FilterChain(QObject *parent)
//	: QObject(parent)
//	, m_validator(nullptr)
//{
//	m_validator = new fc::validation::QtValidator();

//	assert_true() << connect(m_validator, SIGNAL(validationStarted()), this, SIGNAL(validationStarted()))
//				  << connect(m_validator, SIGNAL(validationStep()), this, SIGNAL(validationStep()))
//				  << connect(m_validator, SIGNAL(validationStep()), this, SLOT(onValidationStep()))
//				  << connect(m_validator, SIGNAL(validationComplete()), this, SIGNAL(validationComplete()))
//				  << connect(m_validator, SIGNAL(invalidated()), this, SIGNAL(invalidated()));

//	m_correction = std::make_shared<fc::filter::Correction<float, 2>>();

//	m_postFilterBuffer = fc::filter::makeBuffer(m_correction); // std::make_shared<fc::filter::Buffer<float, 2>>();

//	m_imageOutputChain.setSource(m_postFilterBuffer);

//	m_validator->add(m_imageOutputChain.statistic());
//	m_validator->add(m_imageOutputChain.pixmap());
//}

//FilterChain::~FilterChain()
//{
//	delete m_validator;
//}

//void FilterChain::onValidationStep()
//{
//	if (m_imageOutputChain.pixmap()->isValid())
//		emit pixmapChanged(m_imageOutputChain.pixmap()->data().first());
//}

//void FilterChain::setInput(PyObject *numpy2d)
//{
//#ifdef WITH_THREAD
//	auto filter = std::make_shared<NumpyInput<2>>();
//	filter->setData(numpy2d);
//	m_correction->setPredecessor(filter);
//#else  // WITH_THREAD
//// TODO: Copy numpy-data.
//#endif // WITH_THREAD
//}

//void FilterChain::setInputFitsFile(const QString &filename)
//{
//	auto filter = fc::filter::makeFitsLoader<float, 2>(filename);
//	m_correction->setPredecessor(filter);
//}

//void FilterChain::setDarkImage(PyObject *numpy2d)
//{
//#ifdef WITH_THREAD
//	auto filter = std::make_shared<NumpyInput<2>>();
//	filter->setData(numpy2d);
//	m_correction->setDarkIamge(filter);
//#else  // WITH_THREAD
//// TODO: Copy numpy-data.
//#endif // WITH_THREAD
//}

//void FilterChain::setDarkImages(PyObject *numpy3d, size_t medianDimension)
//{
//#ifdef WITH_THREAD
//	auto filter = std::make_shared<NumpyInput<3>>();
//	filter->setData(numpy3d);
//	auto median = fc::filter::makeMedian("Median of dark images", filter, medianDimension);
//	m_correction->setDarkIamge(median);
//#else  // WITH_THREAD
//// TODO: Copy numpy-data.
//#endif // WITH_THREAD
//}

//void FilterChain::setDarkImageFitsFile(const QString &filename)
//{
//	auto filter = fc::filter::makeFitsLoader<float, 2>(filename);
//	m_correction->setDarkIamge(filter);
//}

//void FilterChain::setDarkImageFitsFiles(const QStringList &filenames)
//{
//	hlp::unused(filenames);
//	// TODO: Not implemented yet!
//}

//void FilterChain::setOpenBeam(PyObject *numpy2d)
//{
//#ifdef WITH_THREAD
//	auto filter = std::make_shared<NumpyInput<2>>();
//	filter->setData(numpy2d);
//	m_correction->setOpenBeam(filter);
//#else  // WITH_THREAD
//// TODO: Copy numpy-data.
//#endif // WITH_THREAD
//}

//void FilterChain::setOpenBeams(PyObject *numpy3d, size_t medianDimension)
//{
//#ifdef WITH_THREAD
//	auto filter = std::make_shared<NumpyInput<3>>();
//	filter->setData(numpy3d);
//	auto median = fc::filter::makeMedian("Median of open beams", filter, medianDimension);
//	m_correction->setOpenBeam(median);
//#else  // WITH_THREAD
//// TODO: Copy numpy-data.
//#endif // WITH_THREAD
//}

//void FilterChain::setOpenBeamFitsFile(const QString &filename)
//{
//	auto filter = fc::filter::makeFitsLoader<float, 2>(filename);
//	m_correction->setOpenBeam(filter);
//}

//void FilterChain::setOpenBeamFitsFiles(const QStringList &filenames)
//{
//	hlp::unused(filenames);
//	// TODO: Not implemented yet!
//}

////void FilterChain::setFilters(const QVector<Skipable2d *> &filterList)
////{
////	// TODO: Not implemented yet!
////}

//// void FilterChain::setFilters(const QList<Skipable2d *> &filterList)
////{
////	std::shared_ptr<const fc::DataFilter<float, 2>> predecessor = m_correction;

////	for (Skipable2d *skipable : filterList) {
////		auto filter = skipable->getSkipableFilter();
////		filter->setPredecessor(std::move(predecessor));
////		predecessor = std::move(filter);
////	}

////	m_postFilterBuffer->setPredecessor(std::move(predecessor));
////}

//void FilterChain::setColorRange(double min, double max)
//{
//	m_imageOutputChain.setColormapRange(std::make_pair(min, max));
//}

//void FilterChain::setUseColor(bool useColor)
//{
//	m_imageOutputChain.setColor(useColor);
//}

//void FilterChain::setInvert(bool invert)
//{
//	m_imageOutputChain.setInvert(invert);
//}

//void FilterChain::setNormalize(bool normalize)
//{
//	// TODO: Not implemented yet!
//}

//void FilterChain::setLogarithmic(bool logarithmic)
//{
//	m_imageOutputChain.setLog(logarithmic);
//}

//bool FilterChain::hasData() const
//{
//	return m_postFilterBuffer->isValid();
//}

//PyObject *FilterChain::data() const
//{
//	// TODO: Not implemented yet!
//	return nullptr;
//}

//bool FilterChain::hasPixmap() const
//{
//	return m_imageOutputChain.pixmap()->isValid();
//}

//QImage FilterChain::pixmap() const
//{
//	if (this->hasPixmap())
//		return this->m_imageOutputChain.pixmap()->data().first();
//	return QImage();
//}

//bool FilterChain::isWorking() const
//{
//	return m_validator->isWorking();
//}

//void FilterChain::start()
//{
//	m_validator->start();
//	//	QImage image(32, 32, QImage::Format_RGB32);
//	//	ndim::pointer<QRgb, 2> imageData;
//	//	imageData.data = hlp::cast_over_void<QRgb *>(image.bits());
//	//	imageData.sizes[0] = 32;
//	//	imageData.sizes[1] = 32;
//	//	imageData.byte_strides[0] = hlp::byte_offset_t(4);
//	//	imageData.byte_strides[1] = hlp::byte_offset_t(image.bytesPerLine());
//	//	for (size_t y = 0; y < 32; ++y)
//	//		for (size_t x = 0; x < 32; ++x)
//	//			imageData(x, y) = qRgb(x * 8, y * 8, (x + y) * 4);
//	//	emit pixmapChanged(QPixmap::fromImage(image));
//}

//void FilterChain::abort(bool wait)
//{
//	m_validator->abort(wait);
//}

//void FilterChain::restart(bool wait)
//{
//	m_validator->restart(wait);
//}

//QStringList FilterChain::stepDescriptions() const
//{
//	// TODO: Not implemented yet!
//	return QStringList();
//}

//void FilterChain::state(size_t &progress, size_t &duration, size_t &step, size_t &stepCount, QString &description) const
//{
//	m_validator->state(progress, duration, step, stepCount, description);
//}
