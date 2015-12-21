#include "ipw/rangeselectwidget.h"
#include "ui_rangeselectwidget.h"

#include <cstddef>
#include <limits>

#include <QVector>

#include <qwt_scale_widget.h>

#include <qwt_picker_machine.h>

#include <QMouseEvent>

#include <cassert>

namespace ipw
{

RangeSelectWidget::RangeSelectWidget(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::RangeSelectWidget)
{
	ui->setupUi(this);

	ui->spinMinMax1->setMinimum(-std::numeric_limits<double>::infinity());
	ui->spinMinMax1->setMaximum(std::numeric_limits<double>::infinity());
	ui->spinMinMax2->setMinimum(-std::numeric_limits<double>::infinity());
	ui->spinMinMax2->setMaximum(std::numeric_limits<double>::infinity());

	ui->histogram->enableAxis(QwtPlot::yLeft, false);
	ui->histogram->axisWidget(QwtPlot::xBottom)->setColorBarEnabled(true);
	ui->histogram->setAxisMaxMajor(QwtPlot::xBottom, 5);
	ui->histogram->setAxisMaxMinor(QwtPlot::xBottom, 9);

	ui->histogram->setCanvasBackground(QBrush(Qt::lightGray));
	ui->histogram->setScaleLog(ui->checkHistogramLog->isChecked());

	ui->histogram->canvas()->installEventFilter(this);

	zoneData = new QwtPlotZoneItem();
	zoneData->setOrientation(Qt::Vertical);
	zoneData->setBrush(QBrush(Qt::white));
	zoneData->attach(ui->histogram);

	zoneSelection = new QwtPlotZoneItem();
	zoneSelection->setOrientation(Qt::Vertical);
	zoneSelection->setBrush(QColor(32, 104, 214, 128));
	zoneSelection->attach(ui->histogram);

	histogram = new QwtPlotHistogram();
	histogram->setPen(QPen(Qt::NoPen));
	histogram->setBrush(QBrush(Qt::black));
	histogram->setBaseline(1);
	histogram->attach(ui->histogram);

	rangePicker = new QwtPlotPicker(QwtPlot::xBottom, QwtPlot::yLeft, ui->histogram->canvas());
	rangePicker->setRubberBand(QwtPicker::RectRubberBand);
	rangePicker->setStateMachine(new QwtPickerDragRectMachine());
	rangePicker->setRubberBandPen(QColor(128, 128, 0));
	connect(rangePicker, SIGNAL(selected(QRectF)), this, SLOT(rangePicked(QRectF)));

	ui->histogram->setAutoReplot(true);
}

RangeSelectWidget::~RangeSelectWidget()
{
	delete ui;
}

void RangeSelectWidget::setStatistic(const QSharedPointer<const ndimdata::DataStatistic> &statistic)
{
	m_statistic = statistic;
	if (!m_statistic)
		return;
	updateHistogram();
	modeChanged();
}

std::pair<double, double> RangeSelectWidget::bounds() const
{
	return std::minmax(m_bound1, m_bound2);
}

double RangeSelectWidget::low() const
{
	return std::min(m_bound1, m_bound2);
}

double RangeSelectWidget::high() const
{
	return std::max(m_bound1, m_bound2);
}

void RangeSelectWidget::setBound1(double bound)
{
	m_bound1 = bound;
	ui->radioRangeCustom->setChecked(true);
	valuesUpdated();
}

void RangeSelectWidget::setBound2(double bound)
{
	m_bound2 = bound;
	ui->radioRangeCustom->setChecked(true);
	valuesUpdated();
}

RangeSelectWidget::RangeMode RangeSelectWidget::rangeMode() const
{
	return ui->radioRangeCustom->isChecked() ? RangeManual : ui->radioRangeAll->isChecked() ? RangeFull : RangeAuto;
}

void RangeSelectWidget::setRangeMode(RangeSelectWidget::RangeMode mode)
{
	switch (mode) {
	case RangeFull:
		ui->radioRangeAll->setChecked(true);
		break;
	case RangeManual:
		ui->radioRangeCustom->setChecked(true);
		break;
	case RangeAuto:
	default:
		ui->radioRangeAuto->setChecked(true);
		break;
	}
}

bool RangeSelectWidget::eventFilter(QObject *object, QEvent *event)
{
	if (object == ui->histogram->canvas() && event->type() == QEvent::MouseButtonRelease) {
		QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event);
		if (mouseEvent->button() == Qt::RightButton) {
			if (ui->radioRangeAuto->isChecked())
				ui->radioRangeAll->setChecked(true);
			else
				ui->radioRangeAuto->setChecked(true);
			if (m_statistic)
				modeChanged();
		}
		return true;
	}
	return QWidget::eventFilter(object, event);
}

void RangeSelectWidget::updateHistogram()
{
	const ndimdata::DataStatistic &statistic(*m_statistic.data());

	const QVector<size_t> bins = QVector<size_t>::fromStdVector(statistic.histogram);
	double min = statistic.min;
	double max = statistic.max;
	double bin_width = (max - min) / bins.size();
	double previous = min;
	double current = min + bin_width;
	QVector<QwtIntervalSample> samples;

	for (int i = 0; i < bins.size(); ++i, previous = current, current += bin_width)
		samples << QwtIntervalSample(bins[i] + std::numeric_limits<double>::epsilon(), previous, current);

	histogram->setSamples(new HistogramSeriesData(samples));

	//	ui->histogram->setAxisScale(QwtPlot::xBottom, min, max);

	zoneData->setInterval(QwtInterval(statistic.min, statistic.max));
}

void RangeSelectWidget::modeChanged()
{
	const ndimdata::DataStatistic &statistic(*m_statistic.data());
	if (ui->radioRangeAll->isChecked()) {
		m_bound1 = statistic.min;
		m_bound2 = statistic.max;
		valuesUpdated();
	} else if (ui->radioRangeAuto->isChecked()) {
		m_bound1 = statistic.auto_low_bound;
		m_bound2 = statistic.auto_high_bound;
		valuesUpdated();
	}
}

double fromSliderValue(int value, int maximum, double min, double max)
{
	return min + double(value) / maximum * (max - min);
}
int toSliderValue(double value, int maximum, double min, double max)
{
	return int((value - min) / (max - min) * maximum);
}

void RangeSelectWidget::valuesUpdated()
{
	updateControls();
	auto pair = std::minmax(m_bound1, m_bound2);
	emit rangeChanged(pair.first, pair.second);
}

void RangeSelectWidget::updateControls()
{
	assert(!m_statistic.isNull());
	const ndimdata::DataStatistic &statistic(*m_statistic.data());
	m_updating = true;
	ui->spinMinMax1->setValue(m_bound1);
	ui->spinMinMax2->setValue(m_bound2);
	ui->sliderMinMax1->setValue(toSliderValue(m_bound1, ui->sliderMinMax1->maximum(), statistic.min, statistic.max));
	ui->sliderMinMax2->setValue(toSliderValue(m_bound2, ui->sliderMinMax1->maximum(), statistic.min, statistic.max));

	auto bounds = std::minmax(m_bound1, m_bound2);
	double diff = bounds.second - bounds.first;
	double min = bounds.first - 4 * diff;
	double max = bounds.second + 4 * diff;
	min = std::max(min, statistic.min);
	max = std::min(max, statistic.max);
	ui->histogram->setAxisScale(QwtPlot::xBottom, min, max);

	zoneSelection->setInterval(QwtInterval(m_bound1, m_bound2).normalized());
	m_updating = false;
}

void RangeSelectWidget::on_sliderMinMax1_valueChanged(int v)
{
	if (m_updating || !m_statistic)
		return;
	ui->radioRangeCustom->setChecked(true);
	m_bound1 = fromSliderValue(v, ui->sliderMinMax1->maximum(), m_statistic->min, m_statistic->max);
	valuesUpdated();
}

void RangeSelectWidget::on_sliderMinMax2_valueChanged(int v)
{
	if (m_updating || !m_statistic)
		return;
	ui->radioRangeCustom->setChecked(true);
	m_bound2 = fromSliderValue(v, ui->sliderMinMax2->maximum(), m_statistic->min, m_statistic->max);
	valuesUpdated();
}

void RangeSelectWidget::on_spinMinMax1_valueChanged(double v)
{
	if (m_updating || !m_statistic)
		return;
	ui->radioRangeCustom->setChecked(true);
	m_bound1 = v;
	valuesUpdated();
}

void RangeSelectWidget::on_spinMinMax2_valueChanged(double v)
{
	if (m_updating || !m_statistic)
		return;
	ui->radioRangeCustom->setChecked(true);
	m_bound2 = v;
	valuesUpdated();
}

void RangeSelectWidget::on_radioRangeAll_clicked()
{
	if (m_updating || !m_statistic)
		return;
	modeChanged();
}

void RangeSelectWidget::on_radioRangeAuto_clicked()
{
	if (m_updating || !m_statistic)
		return;
	modeChanged();
}

void RangeSelectWidget::on_radioRangeCustom_clicked()
{
	if (m_updating || !m_statistic)
		return;
	valuesUpdated();
}

void RangeSelectWidget::rangePicked(QRectF range)
{
	if (m_updating || !m_statistic)
		return;
	ui->radioRangeCustom->setChecked(true);
	m_bound1 = range.left();
	m_bound2 = range.right();
	valuesUpdated();
}

} // namespace hpw
