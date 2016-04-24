#include "tomography.h"

#include <QGLWidget>

#include "reconstructor.h"

#include "threadableglwidget.h"

namespace tomo {

Tomography::Tomography(size_t sinogramResolution, size_t maxAngleCount, float center)
{
    ThreadableGLWidget *widget = new ThreadableGLWidget();
    this->m_glWidget = std::unique_ptr<QWidget>(widget);
    this->m_glWidget->makeCurrent();
    this->m_reconstructor = std::unique_ptr<Reconstructor>(new Reconstructor(*widget->context()));
    this->m_reconstructor->prepare(sinogramResolution, maxAngleCount, center);
    this->m_glWidget->doneCurrent();
}

void Tomography::proc()
{
    std::lock_guard<std::mutex> guard(this->m_mutex);
    while (true) {
        while (!this->m_run) {
            if (this->m_finish)
                return;
            this->m_running_cv.wait(this->m_mutex);
        }
        while (this->m_run) {
            this->m_glWidget->makeCurrent();

            this->m_glWidget->doneCurrent();
        }
    }
}

} // namespace tomo

