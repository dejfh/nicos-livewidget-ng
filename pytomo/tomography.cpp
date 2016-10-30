#include "tomography.h"

#include <tomo/reconstructor.h>
#include <tomo/threadableglwidget.h>

#include "numpy.h"
#include <ndim/numpy.h>

Tomography::Tomography(size_t sinogramResolution, size_t maxAngleCount, float center)
	: tomo::Tomography(sinogramResolution, maxAngleCount, center)
{
}

Tomography::~Tomography()
{
}

void Tomography::onStep()
{
	emit this->stepDone();
}

void Tomography::setOpenBeam(PyObject *openBeam)
{
	hlp::python::Ref openBeamTmp =
		PyArray_FromAny(openBeam, PyArray_DescrFromType(NPY_FLOAT), 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST, nullptr);

	PyArrayObject *openBeamPy = (PyArrayObject *)openBeamTmp.ptr;

	const float *openBeamData = reinterpret_cast<float *>(PyArray_DATA(openBeamPy));

	ndim::pointer<const float, 1> openBeamPtr(openBeamData, ndim::getShape<1>(openBeamPy), ndim::getStrides<1>(openBeamPy));

	tomo::Tomography::setOpenBeam(openBeamPtr);
}

void Tomography::appendSinogram(PyObject *sinogram, PyObject *angles)
{
	hlp::python::Ref sinogramTmp =
		PyArray_FromAny(sinogram, PyArray_DescrFromType(NPY_FLOAT), 2, 2, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST, nullptr);
	hlp::python::Ref anglesTmp = PyArray_FromAny(angles, PyArray_DescrFromType(NPY_FLOAT), 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST, nullptr);
	PyArrayObject *sinogramPy = (PyArrayObject *)sinogramTmp.ptr;
	PyArrayObject *anglesPy = (PyArrayObject *)anglesTmp.ptr;

	const float *sinogramData = reinterpret_cast<float *>(PyArray_DATA(sinogramPy));
	const float *anglesData = reinterpret_cast<float *>(PyArray_DATA(anglesPy));

	ndim::pointer<const float, 2> sinogramPtr(sinogramData, ndim::getShape<2>(sinogramPy), ndim::getStrides<2>(sinogramPy));
	ndim::pointer<const float, 1> anglesPtr(anglesData, ndim::getShape<1>(anglesPy), ndim::getStrides<1>(anglesPy));

	tomo::Tomography::appendSinogram(sinogramPtr, anglesPtr);
}

void Tomography::setReconstruction(PyObject *reconstruction)
{
	hlp::python::Ref reconstructionTmp =
		PyArray_FromAny(reconstruction, PyArray_DescrFromType(NPY_FLOAT), 2, 2, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST, nullptr);
	PyArrayObject *reconstructionPy = (PyArrayObject *)reconstructionTmp.ptr;

	const float *reconstructionData = reinterpret_cast<float *>(PyArray_DATA(reconstructionPy));

	ndim::pointer<const float, 2> reconstructionPtr(reconstructionData, ndim::getShape<2>(reconstructionPy), ndim::getStrides<2>(reconstructionPy));

	tomo::Tomography::setReconstruction(reconstructionPtr);
}

hlp::python::Ref ndimToNumpy(ndim::pointer<const float, 2> data) {
	hlp::python::Gil gil;
	hlp::unused(gil);

	hlp::python::Ref containerRef = ndim::makePyArrayRef(data, gil);

	hlp::python::Ref result = PyArray_FromAny(containerRef.ptr, PyArray_DescrFromType(NPY_FLOAT), 2, 2, NPY_ARRAY_ENSURECOPY, nullptr);

	return result;
}

PyObject *Tomography::getReconstruction()
{
	return ndimToNumpy(tomo::Tomography::getReconstruction().constData()).steal();
}

PyObject *Tomography::getSinogram()
{
	return ndimToNumpy(tomo::Tomography::getSinogram().constData()).steal();
}

PyObject *Tomography::getLikelihood()
{
	return ndimToNumpy(tomo::Tomography::getLikelihood().constData()).steal();
}

PyObject *Tomography::getGradient()
{
	return ndimToNumpy(tomo::Tomography::getGradient().constData()).steal();
}
