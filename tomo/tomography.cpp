#include "tomography.h"

#include "reconstructor.h"

#include "threadableglwidget.h"

namespace tomo
{

Tomography::Tomography(size_t sinogramResolution, size_t maxAngleCount, float center)
	: m_cancel(true)
	, m_makeGuess(true)
	, m_reconstructionFilled(false)
	, m_reconstructionRequested(false)
{
	ThreadableGLWidget *widget = new ThreadableGLWidget();
	this->m_glWidget = std::unique_ptr<QGLWidget>(widget);
	widget->makeCurrent();
	this->m_reconstructor = std::unique_ptr<Reconstructor>(new Reconstructor(widget->context()));
	m_reconstructor->prepare(sinogramResolution, maxAngleCount, center);
	widget->doneCurrent();
}

Tomography::~Tomography()
{
	m_cancel = true;
	if (m_future.valid())
		m_future.wait();
	// children are deleted by unique pointers.
}

void Tomography::setOpenBeam(ndim::pointer<const float, 1> openBeam)
{
	std::unique_lock<std::mutex> lck(m_mutex);
	hlp::unused(lck);
	m_glWidget->makeCurrent();
	m_reconstructor->setOpenBeam(openBeam);
	m_glWidget->doneCurrent();
}

void Tomography::appendSinogram(ndim::pointer<const float, 2> sinogram, ndim::pointer<const float, 1> angles)
{
	std::unique_lock<std::mutex> lck(m_mutex);
	hlp::unused(lck);
	m_glWidget->makeCurrent();
	m_reconstructor->appendSinogram(sinogram, angles);
	m_glWidget->doneCurrent();
}

void Tomography::run()
{
	if (m_future.valid())
		return; // TODO throw exception?
	auto op = [this]() { this->proc(); };
	m_cancel = false;
	m_future = std::async(std::launch::async, op);
}

void Tomography::stop()
{
	m_cancel = true;
	if (m_future.valid())
		m_future.wait();
}

void Tomography::proc()
{
	while (!m_cancel) {
		std::unique_lock<std::mutex> lck(m_mutex);
		hlp::unused(lck);
		m_glWidget->makeCurrent();

		if (m_makeGuess)
			m_reconstructor->guess();
		else
			m_reconstructor->step();

		if (m_reconstructionRequested) {
			int resolution = m_reconstructor->reconstructionResolution();
			m_reconstruction = ndim::makeMutableContainer(ndim::makeSizes(resolution, resolution), &m_reconstruction);
			m_reconstructor->readReconstruction(m_reconstruction.mutableData());
			m_reconstructionFilled = true;
			m_reconstructionRequested = false;
		}

		m_glWidget->doneCurrent();
	}
}

void Tomography::requestReconstruction(ndim::Container<float, 2> *recycle)
{
	if (m_reconstructionRequested)
		return;
	if (recycle)
		m_reconstruction = std::move(*recycle);
	m_reconstructionFilled = false;
	m_reconstructionRequested = true;
}

bool Tomography::reconstructionAvailable() const
{
	return m_reconstructionFilled;
}

ndim::Container<float, 2> Tomography::getReconstruction()
{
	if (!m_reconstructionFilled)
		return ndim::Container<float, 2>(); // TODO throw exception?
	m_reconstructionFilled = false;
	return std::move(m_reconstruction);
}

} // namespace tomo
