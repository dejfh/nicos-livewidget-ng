#include "tomo/threadableglwidget.h"

tomo::ThreadableGLWidget::ThreadableGLWidget(QWidget *parent, const QGLWidget *shareWidget, Qt::WindowFlags f)
	: QGLWidget(parent, shareWidget, f)
{
}

tomo::ThreadableGLWidget::ThreadableGLWidget(QGLContext *context, QWidget *parent, const QGLWidget *shareWidget, Qt::WindowFlags f)
	: QGLWidget(context, parent, shareWidget, f)
{
}

tomo::ThreadableGLWidget::ThreadableGLWidget(const QGLFormat &format, QWidget *parent, const QGLWidget *shareWidget, Qt::WindowFlags f)
	: QGLWidget(format, parent, shareWidget, f)
{
}

tomo::ThreadableGLWidget::~ThreadableGLWidget()
{
}

void tomo::ThreadableGLWidget::paintEvent(QPaintEvent *)
{
}

void tomo::ThreadableGLWidget::resizeEvent(QResizeEvent *)
{
}
