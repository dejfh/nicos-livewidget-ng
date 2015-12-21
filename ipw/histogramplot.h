#ifndef HPW_HISTOGRAMPLOT_H
#define HPW_HISTOGRAMPLOT_H

#include <qwt_plot.h>
#include <QWidget>
#include <qwt_series_data.h>
#include <qwt_samples.h>

namespace ipw
{

class HistogramSeriesData : public QwtArraySeriesData<QwtIntervalSample>
{
    QRectF bounds;

public:
    explicit HistogramSeriesData(QVector<QwtIntervalSample> &samples)
        : QwtArraySeriesData<QwtIntervalSample>(samples)
    {
        typedef QVector<QwtIntervalSample>::const_iterator const_iterator;
        double top = 0.f;
        for (const_iterator it = samples.cbegin(); it != samples.cend(); ++it)
            top = std::max(top, it->value);
        double left = samples.first().interval.minValue();
        double right = samples.last().interval.maxValue();
        bounds = QRectF(left, top, right - left, top);
    }
    virtual ~HistogramSeriesData()
    {
    }

    virtual QRectF boundingRect() const
    {
        return bounds;
    }
};

/**
 * @brief The HistogramPlot class displays a histogram.
 */
class HistogramPlot : public QwtPlot
{
	Q_OBJECT
public:
	explicit HistogramPlot(QWidget *parent = 0);
	~HistogramPlot();

	virtual QSize sizeHint() const;

signals:

public slots:

	void setScaleLog(bool log = true);
};

} // namespace hpw

#endif // HPW_HISTOGRAMPLOT_H
