#ifndef TOMO_TOMOGRAPHY_H
#define TOMO_TOMOGRAPHY_H

#include "ndim/pointer.h"
#include <QScopedPointer>

#include <atomic>
#include <future>
#include <mutex>

#include "ndim/container.h"

class QGLWidget;

namespace tomo
{

class Reconstructor;

class Tomography
{
	QScopedPointer<QGLWidget> m_glWidget;
	QScopedPointer<Reconstructor> m_reconstructor;

	std::future<void> m_future;
	std::atomic<int> m_stepCount;
	std::mutex m_mutex;
	bool m_makeGuess;

	std::atomic<bool> m_forceStep;

	std::atomic<bool> m_reconstructionRequested;
	std::atomic<bool> m_reconstructionFilled;
	ndim::Container<float, 2> m_reconstruction;
	std::atomic<bool> m_sinogramRequested;
	std::atomic<bool> m_sinogramFilled;
	ndim::Container<float, 2> m_sinogram;
	std::atomic<bool> m_likelihoodRequested;
	std::atomic<bool> m_likelihoodFilled;
	ndim::Container<float, 2> m_likelihood;
	std::atomic<bool> m_gradientRequested;
	std::atomic<bool> m_gradientFilled;
	ndim::Container<float, 2> m_gradient;

public:
	Tomography(size_t sinogramResolution, size_t maxAngleCount, float center);
	~Tomography();

	void setOpenBeam(ndim::pointer<const float, 1> openBeam);
	void appendSinogram(ndim::pointer<const float, 2> sinogram, ndim::pointer<const float, 1> angles);
	void setReconstruction(ndim::pointer<const float, 2> reconstruction);

	void run(int stepCount);
	void stop();
	bool running();

private:
	void proc();

protected:
	virtual void onStep() { }

public:
	void setForceSteps(bool forceSteps);
	bool forceSteps();

	void requestReconstruction(ndim::Container<float, 2> *recycle = nullptr);
	bool reconstructionAvailable() const;
	ndim::Container<float, 2> getReconstruction();
	void requestSinogram(ndim::Container<float, 2> *recycle = nullptr);
	bool sinogramAvailable() const;
	ndim::Container<float, 2> getSinogram();
	void requestLikelihood(ndim::Container<float, 2> *recycle = nullptr);
	bool likelihoodAvailable() const;
	ndim::Container<float, 2> getLikelihood();
	void requestGradient(ndim::Container<float, 2> *recycle = nullptr);
	bool gradientAvailable() const;
	ndim::Container<float, 2> getGradient();
};

} // namespace tomo

#endif // TOMO_TOMOGRAPHY_H
