#ifndef TOMO_TOMOGRAPHY_H
#define TOMO_TOMOGRAPHY_H

#include "ndim/pointer.h"
#include <memory>

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
	std::unique_ptr<QGLWidget> m_glWidget;
	std::unique_ptr<Reconstructor> m_reconstructor;

	std::future<void> m_future;
	std::atomic<bool> m_cancel;
	std::mutex m_mutex;
	bool m_makeGuess;

	std::atomic<bool> m_reconstructionFilled;
	std::atomic<bool> m_reconstructionRequested;
	ndim::Container<float, 2> m_reconstruction;

public:
	Tomography(size_t sinogramResolution, size_t maxAngleCount, float center);
	~Tomography();

	void setOpenBeam(ndim::pointer<const float, 1> openBeam);
	void appendSinogram(ndim::pointer<const float, 2> sinogram, ndim::pointer<const float, 1> angles);

	void run();
	void stop();

private:
	void proc();

public:
	void requestReconstruction(ndim::Container<float, 2> *recycle);
	bool reconstructionAvailable() const;
	ndim::Container<float, 2> getReconstruction();
};

} // namespace tomo

#endif // TOMO_TOMOGRAPHY_H
