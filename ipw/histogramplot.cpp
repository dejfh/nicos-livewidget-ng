#include "ipw/histogramplot.h"

#include <qwt/qwt_scale_engine.h>

namespace ipw
{

HistogramPlot::HistogramPlot(QWidget *parent)
    : QwtPlot(parent)
{
}

HistogramPlot::~HistogramPlot() {}

QSize HistogramPlot::sizeHint() const { return QSize(200, 200); }

void HistogramPlot::setScaleLog(bool log)
{
    if (log)
        setAxisScaleEngine(QwtPlot::yLeft, new QwtLogScaleEngine());
    else
        setAxisScaleEngine(QwtPlot::yLeft, new QwtLinearScaleEngine());
}

} // namespace hpw
