#ifndef TOMO_THREADABLEGLWIDGET_H
#define TOMO_THREADABLEGLWIDGET_H

#include <QGLWidget>
#include <QScopedPointer>

namespace tomo
{

class ThreadableGLWidget : public QGLWidget
{
	Q_OBJECT

public:
	explicit ThreadableGLWidget(QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);
	explicit ThreadableGLWidget(QGLContext *context, QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);
	explicit ThreadableGLWidget(const QGLFormat &format, QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);
	~ThreadableGLWidget();

protected:
	virtual void paintEvent(QPaintEvent *);
	virtual void resizeEvent(QResizeEvent *);
};

} // namespace tomo

#endif // TOMO_THREADABLEGLWIDGET_H
