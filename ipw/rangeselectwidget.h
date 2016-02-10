#ifndef HPW_RANGESELECTWIDGET_H
#define HPW_RANGESELECTWIDGET_H

#include <QWidget>
#include <QSharedPointer>

#include <qwt/qwt_plot_histogram.h>
#include <qwt/qwt_plot_zoneitem.h>

#include <qwt/qwt_plot_picker.h>

#include "ndimdata/statistic.h"

namespace ipw
{

namespace Ui
{
class RangeSelectWidget;
}

class RangeSelectWidget : public QWidget
{
	Q_OBJECT

public:
	enum RangeMode { RangeAuto, RangeFull, RangeManual };

	explicit RangeSelectWidget(QWidget *parent = 0);
	~RangeSelectWidget();

	void setStatistic(const QSharedPointer<const ndimdata::DataStatistic> &statistic);

	std::pair<double, double> bounds() const;
	double low() const;
	double high() const;
	void setBound1(double bound);
	void setBound2(double bound);

	RangeMode rangeMode() const;
	void setRangeMode(RangeMode mode);

signals:
	void rangeChanged(double low, double high);

protected:
	virtual bool eventFilter(QObject *object, QEvent *event);

private slots:
	void on_sliderMinMax1_valueChanged(int v);
	void on_sliderMinMax2_valueChanged(int v);
	void on_spinMinMax1_valueChanged(double v);
	void on_spinMinMax2_valueChanged(double v);

	void on_radioRangeAll_clicked();
	void on_radioRangeAuto_clicked();
	void on_radioRangeCustom_clicked();

	void rangePicked(QRectF range);

private:
	QwtPlotHistogram *histogram;
	QwtPlotZoneItem *zoneData;
	QwtPlotZoneItem *zoneSelection;

	QwtPlotPicker *rangePicker;

	QSharedPointer<const ndimdata::DataStatistic> m_statistic;
	bool m_updating;
	double m_bound1;
	double m_bound2;

	void updateHistogram();
	void modeChanged();
	void valuesUpdated();
	void updateControls();

private:
	Ui::RangeSelectWidget *ui;
};

} // namespace hpw

#endif // HPW_RANGESELECTWIDGET_H
