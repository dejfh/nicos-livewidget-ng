#ifndef TOMOGRAPHY_H
#define TOMOGRAPHY_H

#include <QObject>
#include <Python.h>

#include "helper/python/gilhelper.h"

#include <tomo/tomography.h>

class Tomography : public QObject, public tomo::Tomography
{
	Q_OBJECT

public:
	explicit Tomography(size_t sinogramResolution, size_t maxAngleCount, float center);
	~Tomography();

signals:
	void stepDone();

protected:
	virtual void onStep();

public slots:
	void setOpenBeam(PyObject *openBeam);
	void appendSinogram(PyObject *sinogram, PyObject *angles);
	void setReconstruction(PyObject *reconstruction);

	PyObject *getReconstruction();
	PyObject *getSinogram();
	PyObject *getLikelihood();
	PyObject *getGradient();
};

#endif // TOMOGRAPHY_H
