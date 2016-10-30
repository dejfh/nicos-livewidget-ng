
%Module(name=nicos_imageplot, version=1)

%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

%Import ../pyfc/module_pyfc.sip

class ImagePlot : public QWidget
{
%TypeHeaderCode
#include "pyipw.h"
%End
private:
	ImagePlot(const ImagePlot& other);

public:
	ImagePlot(QWidget *parent = 0);

	const QPixmap pixmap() const;
	void setPixmap(const QPixmap &pixmap);
	void setImage(const QImage &image);

	void disableSelection();
	%MethodCode
		sipCpp->setSelectionMode(ImagePlot::SelectionMode::Disable);
	%End
	void setSelectionZoom();
	%MethodCode
		sipCpp->setSelectionMode(ImagePlot::SelectionMode::Zoom);
	%End
	void setSelectionRect();
	%MethodCode
		sipCpp->setSelectionMode(ImagePlot::SelectionMode::Rect);
	%End
	void setSelectionLine();
	%MethodCode
		sipCpp->setSelectionMode(ImagePlot::SelectionMode::Line);
	%End

	void disableGrid();
	%MethodCode
		sipCpp->setGridMode(ImagePlot::GridMode::Disable);
	%End
	void enableGrid();
	%MethodCode
		sipCpp->setGridMode(ImagePlot::GridMode::Auto);
	%End
	void setGridEnabled(bool on);
	%MethodCode
		if (a0)
			sipCpp->setGridMode(ImagePlot::GridMode::Auto);
		else
			sipCpp->setGridMode(ImagePlot::GridMode::Disable);
	%End
	void setGridInterval(double interval);
	%MethodCode
		sipCpp->setGridMode(ImagePlot::GridMode::Custom);
		sipCpp->setGridInterval(a0);
	%End

	QSizeF gridInterval() const;
	void setGridInterval(double interval);

	QRectF selectionRect() const;
	QLineF selectionLine() const;

	void setSelection(QRectF rect);
	void setSelection(QLineF line);
	void setSelection(QPointF start, QPointF end);
	void setSelection(QPointF start, QSizeF size);
	void resetSelection();

	QGraphicsView *graphicsView() const;

public slots:
	void autoScale();
	void setView(QRectF dataRect);
	void setView(QPointF end, QPointF start);
};

class RangeSelectWidget : public QWidget
{
%TypeHeaderCode
#include "pyipw.h"
%End
private:
	RangeSelectWidget(const RangeSelectWidget&);

signals:
	void rangeChanged(double low, double high);

public:
	explicit RangeSelectWidget(QWidget *parent = 0);
	~RangeSelectWidget();

	void setStatistic(DataStatistic statistic);

	double low() const;
	double high() const;

	void setRangeFull();
	%MethodCode
		sipCpp->setRangeMode(RangeSelectWidget::RangeMode::Full);
	%End
	void setRangeAuto();
	%MethodCode
		sipCpp->setRangeMode(RangeSelectWidget::RangeMode::Auto);
	%End
	void setRange(double a, double b);
};
