#ifndef TOMO_THREADABLEGLWIDGET_H
#define TOMO_THREADABLEGLWIDGET_H

#include <QGLWidget>

namespace tomo
{

class ThreadableGLWidget : public QGLWidget
{
public:
	explicit ThreadableGLWidget(QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);
	explicit ThreadableGLWidget(QGLContext *context, QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);
	explicit ThreadableGLWidget(const QGLFormat &format, QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);

	const QGLContext *context() const;

protected:
	virtual void paintEvent(QPaintEvent *);
	virtual void resizeEvent(QResizeEvent *);
};

} // namespace tomo

#endif // TOMO_THREADABLEGLWIDGET_H
