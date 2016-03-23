#ifndef PYFC_FILTERCHAIN_H
#define PYFC_FILTERCHAIN_H

#include <Python.h>

#include <QObject>
#include <QVector>
#include <QPixmap>

#include "fc/validation/qtvalidator.h"

#include "fc/chains/pixmapoutput.h"

namespace fc
{
namespace filter
{

template <typename T, size_t D>
class Correction;
template <typename T, size_t D>
class Buffer;

} // namespace filter
} // namespace fc

class FilterChain : public QObject
{
	Q_OBJECT

	fc::validation::QtValidator *m_validator;

	std::shared_ptr<fc::filter::Correction<float, 2>> m_correction;
	std::shared_ptr<fc::filter::Buffer<float, 2>> m_postFilterBuffer;

	fc::chains::ImageOutputChain m_imageOutputChain;

public:
	explicit FilterChain(QObject *parent = 0);
	~FilterChain();

private slots:
	void onValidationStep();

signals:
	void validationStarted();
	void validationStep();
	void validationComplete();
	void invalidated();

	void dataChanged();
	void pixmapChanged(QImage pixmap);
	void statisticChanged();

public slots:
	void setInput(PyObject *numpy2d);
	void setInputFitsFile(const QString &filename);

	void setDarkImage(PyObject *numpy2d);
	void setDarkImages(PyObject *numpy3d, size_t medianDimension);
	void setDarkImageFitsFile(const QString &filename);
	void setDarkImageFitsFiles(const QStringList &filenames);

	void setOpenBeam(PyObject *numpy2d);
	void setOpenBeams(PyObject *numpy3d, size_t medianDimension);
	void setOpenBeamFitsFile(const QString &filename);
	void setOpenBeamFitsFiles(const QStringList &filenames);

//	void setFilters(const QVector<Skipable2d *> &filterList);

	void setUseColor(bool useColor);
	void setInvert(bool invert);
	void setNormalize(bool normalize);
	void setLogarithmic(bool logarithmic);

	void setColorRange(double min, double max);

public:
	bool hasData() const;
	PyObject *data() const;
	bool hasPixmap() const;
	QImage pixmap() const;

	bool isWorking() const;
	void start();
	void abort(bool wait);
	void restart(bool wait);

	QStringList stepDescriptions() const;
	void state(size_t &progress, size_t &duration, size_t &step, size_t &stepCount, QString &description) const;
};

#endif // PYFC_FILTERCHAIN_H
