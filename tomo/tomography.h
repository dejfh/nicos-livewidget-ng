#ifndef TOMO_TOMOGRAPHY_H
#define TOMO_TOMOGRAPHY_H

#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "ndim/pointer.h"

#include "helper/threadsafe.h"

class QGLWidget;

namespace tomo {

class Reconstructor;

class Tomography
{
public:
    Tomography(size_t sinogramResolution, size_t maxAngleCount, float center);

    void setDarkImage(ndim::pointer<const float, 1> darkImage);
    void setOpenBeam(ndim::pointer<const float, 1> openBeam);
    void appendSinogram( ndim::pointer<const std::int16_t, 2> sinogram, ndim::pointer<const float, 1> angles);

    void run();
    void stop();

    void requestReconstruction();
    void getReconstruction(ndim::pointer<float, 2> data);

private slots:
    void proc();

private:
    std::unique_ptr<QGLWidget> m_glWidget;
    std::unique_ptr<Reconstructor> m_reconstructor;
    std::mutex m_mutex;
    std::atomic<bool> m_run;
    std::condition_variable m_running_cv;
    std::atomic<bool> m_finish;
};

} // namespace tomo

#endif // TOMO_TOMOGRAPHY_H
