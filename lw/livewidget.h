#ifndef LW_LIVEWIDGET_H
#define LW_LIVEWIDGET_H

#include <QMainWindow>
#include <QScopedPointer>
#include <QStringList>
#include <QMap>

#include "ipw/imageplot.h"

#include "lw/livewidgetchain.h"

#include "ndim/range.h"

class QDir;
class QLabel;
class QProgressBar;
class QListWidgetItem;

namespace lw
{

namespace Ui
{
class LiveWidget;
}

class LiveWidget : public QMainWindow
{
	Q_OBJECT

public:
	explicit LiveWidget(const QDir &location, QWidget *parent = 0);
	~LiveWidget();

	ndim::range<2> roi() const;
	bool hasRoi() const;
	void resetRoi();
	void setRoi(ndim::range<2> roi);

private slots:
	void updateProgress();

	void setValidationParameter();

	void on_comboKeyword_currentTextChanged(const QString &keyword);

	void setSelectionMode(ipw::ImagePlot::SelectionMode mode);

	void on_actionZoom_triggered();
	void on_actionLine_triggered();
	void on_actionRectangle_triggered();
	void on_plot_selectionChanged(QRectF selection);
	void on_plot_selectionChanged(QLineF selection);
	void on_actionResetView_triggered();
	void on_checkGrid_clicked();

	void on_actionResetSelection_triggered();

	void on_actionPlotProfile_triggered();
	void on_actionSetRoi_triggered();
	void on_actionZStackPlot_triggered();
	void on_listZStackFiles_itemSelectionChanged();

	void on_listSourceFiles_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous);
	void on_radioSourceImage_clicked();
	void on_radioSourceOpenBeam_clicked();
	void on_radioSourceDarkImage_clicked();
	void on_checkColor_clicked();
	void on_checkInvert_clicked();
	void on_checkNormalize_clicked();
	void on_checkLogarithmic_clicked();
	void on_range_rangeChanged(double, double);

	void on_buttonRoiSync_clicked(bool checked);

	void on_buttonRoiReset_clicked();

private:
	Ui::LiveWidget *ui;

	bool m_updating;

	QLabel *m_progressLabel;
	QProgressBar *m_progressBar;

	QStringList m_filenames;
	QMap<QString, QStringList> m_keywords;

	LiveWidgetChain m_filterChain;
};

} // namespace lw

#endif // LW_LIVEWIDGET_H
