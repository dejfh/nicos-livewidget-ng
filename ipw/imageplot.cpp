#include "ipw/imageplot.h"
#include "ui_imageplot.h"

#include <limits>
#include <algorithm>

#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QGraphicsLineItem>

#include <QWheelEvent>

#include "ipw/fixedratiotransform.h"
#include "ipw/griditem.h"
#include "ipw/trackeritem.h"
#include "ipw/selectionrectitem.h"
#include "ipw/selectionlineitem.h"

#include "helper/helper.h"

#include "safecast.h"

namespace ipw
{

ImagePlot::ImagePlot(QWidget *parent)
	: QWidget(parent)
	, m_scene(new QGraphicsScene(this))
	, m_transform(new FixedRatioTransform(this))
	, m_pixmapItem(0)
	, m_trackerItem(0)
	, m_selectionRectItem(0)
	, m_selectionLineItem(0)
	, m_selectionMode(SelectionZoom)
	, m_validSelection(false)
	, ui(new Ui::ImagePlot)
{
	ui->setupUi(this);

	int lineWidth = ui->view->lineWidth();
	ui->scaleLeft->setAlignment(QwtScaleDraw::LeftScale);
	ui->scaleLeft->setBorderDist(lineWidth, lineWidth);
	ui->scaleBottom->setAlignment(QwtScaleDraw::BottomScale);
	ui->scaleBottom->setBorderDist(lineWidth, lineWidth);

	m_pixmapItem = new QGraphicsPixmapItem();
	m_scene->addItem(m_pixmapItem);

	m_gridItem = new GridItem(m_transform);
	m_scene->addItem(m_gridItem);

	m_trackerItem = new TrackerItem(m_transform);
	m_trackerItem->setHoverLookup([=](QPointF dataPoint) { return this->hoverText(dataPoint); });
	m_scene->addItem(m_trackerItem);

	m_selectionRectItem = new SelectionRectItem(m_transform);
	m_selectionRectItem->setVisible(false);
	m_scene->addItem(m_selectionRectItem);

	m_selectionLineItem = new SelectionLineItem(m_transform);
	m_selectionLineItem->setVisible(false);
	m_scene->addItem(m_selectionLineItem);

	ui->view->setSceneRect(0, 0, 1, 1);
	ui->view->setRenderHint(QPainter::Antialiasing);
	ui->view->setScene(m_scene);

	hlp::assert_true() << connect(m_trackerItem, SIGNAL(rightClick(QPointF)), this, SLOT(autoScale()));
	hlp::assert_true() << connect(m_trackerItem, SIGNAL(selectionComplete(QRectF)), this, SLOT(makeSelection(QRectF)));
	hlp::assert_true() << connect(m_transform, SIGNAL(viewChanged()), this, SLOT(updatePixmapGeometry()));
	hlp::assert_true() << connect(m_transform, SIGNAL(viewChanged()), this, SLOT(updateScales()));
	hlp::assert_true() << connect(this, SIGNAL(resized()), this, SLOT(updateScales()), Qt::QueuedConnection);
	hlp::assert_true() << connect(m_selectionRectItem, SIGNAL(selectionChanged(QRectF)), this, SLOT(selectionItemChanged(QRectF)));
	hlp::assert_true() << connect(m_selectionRectItem, SIGNAL(rightClicked(QRectF, QPoint)), this, SLOT(selectionItemRightClicked(QRectF, QPoint)));
	hlp::assert_true() << connect(m_selectionLineItem, SIGNAL(selectionChanged(QLineF)), this, SLOT(selectionItemChanged(QLineF)));
	hlp::assert_true() << connect(m_selectionLineItem, SIGNAL(rightClicked(QLineF, QPoint)), this, SLOT(selectionItemRightClicked(QLineF, QPoint)));

	ui->view->installEventFilter(this);
}

ImagePlot::~ImagePlot()
{
}

const QPixmap ImagePlot::pixmap() const
{
	return m_pixmapItem->pixmap();
}

void ImagePlot::setPixmap(const QPixmap &pixmap)
{
	m_pixmapItem->setPixmap(pixmap);
	m_transform->setDataRect(QRectF(QPointF(), pixmap.size()));
}

std::function<double(size_t, size_t)> ImagePlot::valueLookup() const
{
	return m_valueLookup;
}

void ImagePlot::setValueLookup(std::function<double(size_t, size_t)> valueLookup)
{
	m_valueLookup = valueLookup;
}

ImagePlot::SelectionMode ImagePlot::selectionMode() const
{
	return m_selectionMode;
}

void ImagePlot::setSelectionMode(ImagePlot::SelectionMode mode)
{
	m_selectionMode = mode;
	switch (mode) {
	case SelectionZoom:
		m_trackerItem->setSelectionMode(TrackerItem::RectSelection);
		m_selectionRectItem->setVisible(false);
		m_selectionLineItem->setVisible(false);
		break;
	case SelectionRect:
		m_trackerItem->setSelectionMode(TrackerItem::RectSelection);
		m_selectionRectItem->setVisible(m_validSelection);
		m_selectionLineItem->setVisible(false);
		break;
	case SelectionLine:
		m_trackerItem->setSelectionMode(TrackerItem::LineSelection);
		m_selectionRectItem->setVisible(false);
		m_selectionLineItem->setVisible(m_validSelection);
		break;
	default:
		m_trackerItem->setSelectionMode(TrackerItem::DisableSelection);
		m_selectionRectItem->setVisible(false);
		m_selectionLineItem->setVisible(false);
		break;
	}
}

ImagePlot::GridMode ImagePlot::gridMode() const
{
	if (!m_gridItem->isVisible())
		return GridDisable;
	else if (m_gridItem->useManualInterval())
		return GridCustom;
	else
		return GridAuto;
}

void ImagePlot::setGridMode(ImagePlot::GridMode mode)
{
	switch (mode) {
	case GridAuto:
		m_gridItem->setUseManualInterval(false);
		m_gridItem->setVisible(true);
		break;
	case GridCustom:
		m_gridItem->setUseManualInterval(true);
		m_gridItem->setVisible(true);
		break;
	default:
		m_gridItem->setUseManualInterval(false);
		m_gridItem->setVisible(false);
		break;
	}
	updateScales();
}

QSizeF ImagePlot::gridInterval() const
{
	return m_gridItem->manualInterval();
}

void ImagePlot::setGridInterval(double interval)
{
	m_gridItem->setManualInterval(interval);
	updateScales();
}

QRectF ImagePlot::selectionRect() const
{
	return m_selectionRectItem->rect();
}

QLineF ImagePlot::selectionLine() const
{
	return m_selectionLineItem->line();
}

void ImagePlot::setSelection(QRectF rect)
{
	m_selectionRectItem->setRect(rect);
	m_selectionLineItem->setLine(QLineF(rect.topLeft(), rect.bottomRight()));
	m_validSelection = true;
	m_selectionRectItem->setVisible(m_selectionMode == SelectionRect);
	m_selectionLineItem->setVisible(m_selectionMode == SelectionLine);
}

void ImagePlot::setSelection(QLineF line)
{
	setSelection(QRectF(line.p1(), line.p2()));
}

void ImagePlot::setSelection(QPointF start, QPointF end)
{
	setSelection(QRectF(start, end));
}

void ImagePlot::setSelection(QPointF start, QSizeF size)
{
	setSelection(QRectF(start, size));
}

void ImagePlot::resetSelection()
{
	m_selectionRectItem->setVisible(false);
	m_selectionLineItem->setVisible(false);
}

QGraphicsView *ImagePlot::graphicsView() const
{
	return ui->view;
}

void ImagePlot::autoScale()
{
	setView(m_transform->dataRect());
}

QwtScaleDiv makeScaleDiv(double min, double start, double interval, double max)
{
	QwtScaleDiv scaleDiv(min, max);
	QList<double> ticks;

	for (; start < max; start += interval)
		ticks.append(start);
	scaleDiv.setTicks(QwtScaleDiv::MajorTick, ticks);
	return scaleDiv;
}

void ImagePlot::updateScales()
{
	QRect widgetRect = ui->view->contentsRect();
	{
		// ensure that scene pixels are exactly on widget pixels
		QRectF centerRect(-1, -1, 1, 1);
		if (widgetRect.width() % 2 == 0)
			centerRect.setRight(1);
		if (widgetRect.height() % 2 == 0)
			centerRect.setBottom(1);
		ui->view->setSceneRect(centerRect);
	}

	QRectF sceneRect(ui->view->mapToScene(widgetRect.topLeft()), ui->view->mapToScene(widgetRect.bottomRight()));
	QRectF dataRect(m_transform->sceneToData(sceneRect).normalized());
	QRectF firstInterval = m_gridItem->firstDataInterval(dataRect);

	ui->scaleBottom->setScaleDiv(makeScaleDiv(dataRect.left(), firstInterval.left(), firstInterval.width(), dataRect.right()));
	ui->scaleLeft->setScaleDiv(makeScaleDiv(dataRect.top(), firstInterval.top(), firstInterval.height(), dataRect.bottom()));
	ui->scaleBottom->setBorderDist(1, 1);
	ui->scaleLeft->setBorderDist(1, 1);
}

void ImagePlot::makeSelection(QRectF rect)
{
	QLineF line(rect.topLeft(), rect.bottomRight());
	switch (m_selectionMode) {
	case SelectionZoom:
		setView(rect);
		break;
	case SelectionRect:
		setSelection(rect);
		selectionChanged(rect);

		break;
	case SelectionLine:
		setSelection(rect);
		selectionChanged(line);
		break;
	default:
		break;
	}
}

void ImagePlot::selectionItemChanged(QRectF selection)
{
	emit selectionChanged(selection);
}

void ImagePlot::selectionItemChanged(QLineF selection)
{
	emit selectionChanged(selection);
}

void ImagePlot::selectionItemRightClicked(QRectF selection, QPoint point)
{
	emit selectionRightClicked(selection, point);
}

void ImagePlot::selectionItemRightClicked(QLineF selection, QPoint point)
{
	emit selectionRightClicked(selection, point);
}

QString ImagePlot::hoverText(QPointF dataPoint)
{
	QPoint roundedPoint(int(std::floor(dataPoint.x())), int(std::floor(dataPoint.y())));
	try {
		if (m_valueLookup && m_transform->dataRect().toRect().contains(roundedPoint))
			return QString("x: %1 y: %2\nvalue: %3")
				.arg(roundedPoint.x())
				.arg(roundedPoint.y())
				.arg(m_valueLookup(roundedPoint.x(), roundedPoint.y()));
	} catch (...) {
	}
	return QString("x: %1 y: %2").arg(roundedPoint.x()).arg(roundedPoint.y());
}

void ImagePlot::setView(QRectF dataRect)
{
	dataRect = dataRect.normalized();
	QSize viewSize = ui->view->contentsRect().size();
	double zoom = std::min(viewSize.width() / dataRect.width(), viewSize.height() / dataRect.height());
	m_transform->setView(zoom, dataRect.center(), QPointF());
}

void ImagePlot::setView(QPointF start, QPointF end)
{
	if (start != end)
		setView(QRectF(start, end));
	else {
		QPointF scenePoint = m_transform->dataToScene(start);
		m_transform->setZoom(m_transform->zoom() * std::pow(2, 0.5), scenePoint);
	}
}

bool ImagePlot::eventFilter(QObject *object, QEvent *event)
{
	if (object == ui->view) {
		switch (event->type()) {
		case QEvent::Wheel: {
			QWheelEvent *wheelEvent = static_cast<QWheelEvent *>(event);
			QPointF scenePoint = ui->view->mapToScene(wheelEvent->pos());
#if QT_VERSION >= 0x050000
			double delta = double(wheelEvent->angleDelta().y()) / (120 * 4);
#else
			double delta = double(wheelEvent->delta()) / (120 * 4);
#endif
			m_transform->setZoom(m_transform->zoom() * std::pow(2, double(delta)), scenePoint);
			return true;
		}
		case QEvent::Resize:
			emit resized();
			break;
		default:
			break;
		}
	}
	return QObject::eventFilter(object, event);
}

void ImagePlot::updatePixmapGeometry()
{
	m_pixmapItem->setScale(m_transform->zoom());
	m_pixmapItem->setPos(m_transform->dataToScene(m_transform->dataRect().bottomLeft()));
}

} // namespace hpw
