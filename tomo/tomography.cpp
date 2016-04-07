#include "tomography.h"

#include "reconstructor.h"

#include "threadableglwidget.h"

namespace tomo {

Tomography::Tomography(size_t sinogramResolution, size_t maxAngleCount, size_t reconstructionResolution)
{
    ThreadableGLWidget *widget = new ThreadableGLWidget();
    this->m_glWidget = widget;
    widget->
}

} // namespace tomo

