#ifndef HPW_HEATMAPPLOT_H
#define HPW_HEATMAPPLOT_H

#include <functional>

#include <QWidget>
#include <QRectF>
#include <QPointF>
#include <QSizeF>
#include <QLineF>

#include <QPoint>

#include <QGraphicsView>

class QGraphicsScene;
class QGraphicsPixmapItem;

namespace ipw
{
class FixedRatioTransform;
class GridItem;
class TrackerItem;
class SelectionRectItem;
class SelectionLineItem;

namespace Ui
{
class ImagePlot;
}

class ImagePlot : public QWidget
{
	Q_OBJECT

public:
	enum SelectionMode { SelectionDisable, SelectionZoom, SelectionRect, SelectionLine };
	enum GridMode { GridDisable, GridAuto, GridCustom };

	explicit ImagePlot(QWidget *parent = 0);
	~ImagePlot();

	const QPixmap pixmap() const;
	std::function<double(size_t, size_t)> valueLookup() const;
	SelectionMode selectionMode() const;

	GridMode gridMode() const;
	QSizeF gridInterval() const;
	QRectF selectionRect() const;
	QLineF selectionLine() const;

	QGraphicsView *graphicsView() const;

public slots:
	void setPixmap(const QPixmap &pixmap);
	void setImage(const QImage &image);
	void setValueLookup(std::function<double(size_t, size_t)> valueLookup);
	void setSelectionMode(SelectionMode mode);
	void setGridMode(GridMode mode);
	void setGridInterval(double interval);

public slots:
	void setSelection(QRectF rect);
	void setSelection(QLineF line);
	void setSelection(QPointF start, QPointF end);
	void setSelection(QPointF start, QSizeF size);
	void resetSelection();

public slots:
	void autoScale();
	void setView(QRectF dataRect);
	void setView(QPointF end, QPointF start);

signals:
	void resized();
	void selectionChanged(QRectF selection);
	void selectionChanged(QLineF selection);
	void selectionRightClicked(QRectF selection, QPoint point);
	void selectionRightClicked(QLineF selection, QPoint point);

protected:
	virtual bool eventFilter(QObject *object, QEvent *event);

private slots:
	void updatePixmapGeometry();
	void updateScales();
	void makeSelection(QRectF selection);
	void selectionItemChanged(QRectF selection);
	void selectionItemChanged(QLineF selection);
	void selectionItemRightClicked(QRectF selection, QPoint point);
	void selectionItemRightClicked(QLineF selection, QPoint point);

private:
	QString hoverText(QPointF dataPoint);

	QGraphicsScene *m_scene;
	FixedRatioTransform *m_transform;

	QGraphicsPixmapItem *m_pixmapItem;
	GridItem *m_gridItem;
	TrackerItem *m_trackerItem;
	SelectionRectItem *m_selectionRectItem;
	SelectionLineItem *m_selectionLineItem;

	SelectionMode m_selectionMode;
	bool m_validSelection;

	std::function<double(size_t, size_t)> m_valueLookup;

	Ui::ImagePlot *ui;
};

} // namespace hpw

#endif // HPW_HEATMAPPLOT_H
