#include "tomography.h"

#include "reconstructor.h"

#include "threadableglwidget.h"

namespace tomo
{

struct GLContextLock {
	QGLWidget &m_glWidget;

	GLContextLock(QGLWidget &glWidget)
		: m_glWidget(glWidget)
	{
		m_glWidget.makeCurrent();
	}
	GLContextLock(const GLContextLock &) = delete;
	GLContextLock(GLContextLock &&) = delete;
	~GLContextLock()
	{
		m_glWidget.doneCurrent();
	}
};

Tomography::Tomography(size_t sinogramResolution, size_t maxAngleCount, float center)
	: m_stepCount(0)
	, m_makeGuess(true)
	, m_forceStep(false)
	, m_reconstructionRequested(false)
	, m_reconstructionFilled(false)
	, m_sinogramRequested(false)
	, m_sinogramFilled(false)
	, m_likelihoodRequested(false)
	, m_likelihoodFilled(false)
	, m_gradientRequested(false)
	, m_gradientFilled(false)
{
	ThreadableGLWidget *widget = new ThreadableGLWidget();
	this->m_glWidget.reset(widget);
	GLContextLock glLck(*widget);
	hlp::unused(glLck);

	this->m_reconstructor.reset(new Reconstructor(widget->context()));
	m_reconstructor->prepare(sinogramResolution, maxAngleCount, center);
}

Tomography::~Tomography()
{
	m_stepCount = 0;
	if (m_future.valid())
		m_future.wait();
	// children are deleted by scoped pointers.
}

void Tomography::setOpenBeam(ndim::pointer<const float, 1> openBeam)
{
	std::unique_lock<std::mutex> lck(m_mutex);
	GLContextLock glLck(*m_glWidget);
	hlp::unused(lck, glLck);

	m_reconstructor->setOpenBeam(openBeam);
}

void Tomography::appendSinogram(ndim::pointer<const float, 2> sinogram, ndim::pointer<const float, 1> angles)
{
	std::unique_lock<std::mutex> lck(m_mutex);
	GLContextLock glLck(*m_glWidget);
	hlp::unused(lck, glLck);

	m_reconstructor->appendSinogram(sinogram, angles);
}

void Tomography::setReconstruction(ndim::pointer<const float, 2> reconstruction)
{
	std::unique_lock<std::mutex> lck(m_mutex);
	GLContextLock glLck(*m_glWidget);
	hlp::unused(lck, glLck);

	m_reconstructor->setReconstruction(reconstruction);
}

void Tomography::run(int stepCount)
{
	if (running())
		return; // TODO throw exception?
	auto op = [this]() { this->proc(); };
	m_stepCount = stepCount;
	m_future = std::async(std::launch::async, op);
}

void Tomography::stop()
{
	m_stepCount = 0;
	if (m_future.valid())
		m_future.wait();
}

bool Tomography::running()
{
	if (!m_future.valid())
		return false;
	std::future_status result = m_future.wait_for(std::chrono::seconds::zero());
	if (result == std::future_status::timeout)
		return true;
	return false;
}

void Tomography::proc()
{
	while (m_stepCount-- != 0) {
		std::unique_lock<std::mutex> lck(m_mutex);
		GLContextLock glLck(*m_glWidget);
		hlp::unused(lck, glLck);

		if (m_makeGuess) {
			m_reconstructor->guess();
			m_makeGuess = false;
		} else {
			m_reconstructor->step(m_forceStep);
		}

		int resolution = m_reconstructor->reconstructionResolution();
		int resolutionS = m_reconstructor->sinogramResolution();
		int count = m_reconstructor->sinogramFilled();

		if (m_reconstructionRequested) {
			m_reconstruction = ndim::makeMutableContainer(ndim::makeSizes(resolution, resolution), &m_reconstruction);
			m_reconstructor->readTexture(m_reconstruction.mutableData(), Reconstructor::TexRecon);
			m_reconstructionFilled = true;
			m_reconstructionRequested = false;
		}
		if (m_sinogramRequested) {
			m_sinogram = ndim::makeMutableContainer(ndim::makeSizes(resolutionS, count), &m_sinogram);
			m_reconstructor->readTexture(m_sinogram.mutableData(), Reconstructor::TexSinoRecon);
			m_sinogramFilled = true;
			m_sinogramRequested = false;
		}
		if (m_likelihoodRequested) {
			m_likelihood = ndim::makeMutableContainer(ndim::makeSizes(resolutionS, count), &m_likelihood);
			m_reconstructor->readTexture(m_likelihood.mutableData(), Reconstructor::TexLikelihood);
			m_likelihoodFilled = true;
			m_likelihoodRequested = false;
		}
		if (m_gradientRequested) {
			m_gradient = ndim::makeMutableContainer(ndim::makeSizes(resolution, resolution), &m_gradient);
			m_reconstructor->readTexture(m_gradient.mutableData(), Reconstructor::TexGradient);
			m_gradientFilled = true;
			m_gradientRequested = false;
		}

		this->onStep();
	}
}

void Tomography::setForceSteps(bool forceSteps)
{
	m_forceStep = forceSteps;
}

bool Tomography::forceSteps()
{
	return m_forceStep;
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

void Tomography::requestSinogram(ndim::Container<float, 2> *recycle)
{
	if (m_sinogramRequested)
		return;
	if (recycle)
		m_sinogram = std::move(*recycle);
	m_sinogramFilled = false;
	m_sinogramRequested = true;
}

bool Tomography::sinogramAvailable() const
{
	return m_sinogramFilled;
}

ndim::Container<float, 2> Tomography::getSinogram()
{
	if (!m_sinogramFilled)
		return ndim::Container<float, 2>(); // TODO throw exception?
	m_sinogramFilled = false;
	return std::move(m_sinogram);
}

void Tomography::requestLikelihood(ndim::Container<float, 2> *recycle)
{
	if (m_likelihoodRequested)
		return;
	if (recycle)
		m_likelihood = std::move(*recycle);
	m_likelihoodFilled = false;
	m_likelihoodRequested = true;
}

bool Tomography::likelihoodAvailable() const
{
	return m_likelihoodFilled;
}

ndim::Container<float, 2> Tomography::getLikelihood()
{
	if (!m_likelihoodFilled)
		return ndim::Container<float, 2>(); // TODO throw exception?
	m_likelihoodFilled = false;
	return std::move(m_likelihood);
}

void Tomography::requestGradient(ndim::Container<float, 2> *recycle)
{
	if (m_gradientRequested)
		return;
	if (recycle)
		m_gradient = std::move(*recycle);
	m_gradientFilled = false;
	m_gradientRequested = true;
}

bool Tomography::gradientAvailable() const
{
	return m_gradientFilled;
}

ndim::Container<float, 2> Tomography::getGradient()
{
	if (!m_gradientFilled)
		return ndim::Container<float, 2>(); // TODO throw exception?
	m_gradientFilled = false;
	return std::move(m_gradient);
}

} // namespace tomo
