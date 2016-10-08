#ifndef TOMO_THREADABLEGLWIDGET_H
#define TOMO_THREADABLEGLWIDGET_H

#include <QGLWidget>
#include <QScopedPointer>

namespace tomo
{

class ThreadableGLWidget : public QGLWidget
{
	Q_OBJECT

private:
	QGLContext *m_context;

public:
	explicit ThreadableGLWidget(QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);
	explicit ThreadableGLWidget(QGLContext *context, QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);
	explicit ThreadableGLWidget(const QGLFormat &format, QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0);

protected:
	virtual void paintEvent(QPaintEvent *);
	virtual void resizeEvent(QResizeEvent *);

public:
	QGLContext &context()
	{
		return *m_context;
	}

	static ThreadableGLWidget *create(QWidget *parent = 0, const QGLWidget *shareWidget = 0, Qt::WindowFlags f = 0)
	{
		QGLFormat format = QGLFormat::defaultFormat();
		QGLContext *context = new QGLContext(format);
		try {
			ThreadableGLWidget *widget = new ThreadableGLWidget(context, parent, shareWidget, f);
			widget->m_context = context;
			return widget;
		} catch (...) {
			if (context) {
				delete context;
			}
		}
	}
};

} // namespace tomo

#endif // TOMO_THREADABLEGLWIDGET_H
