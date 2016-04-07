#ifndef TOMO_TOMOGRAPHY_H
#define TOMO_TOMOGRAPHY_H

#include "ndim/pointer.h"
#include <memory>

class QWidget;

namespace tomo {

class Reconstructor;

class Tomography
{
public:
    Tomography(size_t sinogramResolution, size_t maxAngleCount, size_t reconstructionResolution);

    void setDarkImage(ndim::pointer<const float, 1> darkImage);
    void setOpenBeam(ndim::pointer<const float, 1> openBeam);
    void appendSinogram(ndim::pointer<const std::int16_t, 2> sinogram, ndim::pointer<const float, 1> angles);

    void run();
    void stop();

    void requestReconstruction();
    void getReconstruction(ndim::pointer<float, 2> data);

private:
    std::unique_ptr<QWidget> m_glWidget;
    std::unique_ptr<Reconstructor> m_reconstructor;
};

} // namespace tomo

#endif // TOMO_TOMOGRAPHY_H
