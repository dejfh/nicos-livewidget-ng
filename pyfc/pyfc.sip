
%Module(name=nicosfilterchain, version=1)

%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

%ModuleHeaderCode
void init_pyfc();
%End

%InitialisationCode
init_pyfc();
%End

class Filter2d
{
%TypeHeaderCode
#include "filter2d.h"
%End

private:
	Filter2d();
};

class Skipable2d : public Filter2d
{
%TypeHeaderCode
#include "skipable2d.h"
%End

private:
	Skipable2d();
};

class NoopFilter2d : public Skipable2d
{
%TypeHeaderCode
#include "noopfilter2d.h"
%End

public:
	NoopFilter2d();
};

class FilterChain : public QObject
{
%TypeHeaderCode
#include "filterchain.h"
%End

public:
	explicit FilterChain(QObject *parent = 0);

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
	void setDarkImages(PyObject *numpy3d, quint32 medianDimension);
	void setDarkImageFitsFile(const QString &filename);
	void setDarkImageFitsFiles(const QStringList &filenames);

	void setOpenBeam(PyObject *numpy2d);
	void setOpenBeams(PyObject *numpy3d, quint32 medianDimension);
	void setOpenBeamFitsFile(const QString &filename);
	void setOpenBeamFitsFiles(const QStringList &filenames);

	//void setFilters(const QVector<Skipable2d *> &filterList);

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
// void state(quint64 &progress, quint64 &duration, quint64 &step, quint64 &stepCount, QString &description) const;
};
