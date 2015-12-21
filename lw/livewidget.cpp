#include "livewidget.h"
#include "ui_livewidget.h"

#include <QDir>
#include <QProgressBar>
#include <QLabel>
#include <QHBoxLayout>
#include <QString>
#include <QTimer>

#include <QVector>
#include <QListWidgetItem>
#include <QModelIndex>

#include "tw/projectlocation.h"

#include "fits/fitsloader.h"

#include "safecast.h"

#include "qwt_plot_curve.h"

using jfh::assert_result;

namespace lw
{

LiveWidget::LiveWidget(const QDir &location, QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::LiveWidget)
	, m_updating(false)

	, m_progressLabel(0)
	, m_progressBar(0)
{
	QDir::setCurrent(location.absolutePath());

	ui->setupUi(this);
	ui->dockProfile->setVisible(false);
	ui->dockRoi->setVisible(false);
	ui->dockZStackSource->setVisible(false);
	ui->dockZStackPlot->setVisible(false);

	ui->plotProfile->setCanvasBackground(Qt::white);
	ui->plotZStack->setCanvasBackground(Qt::white);

	m_progressLabel = new QLabel();
	m_progressBar = new QProgressBar();
	m_progressBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Ignored);
	m_progressBar->setTextVisible(false);
	m_progressBar->setMaximum(10000);
	m_progressBar->setVisible(false);
	ui->statusbar->addWidget(m_progressLabel);
	ui->statusbar->addWidget(m_progressBar, 1);

	ui->menuLine->setEnabled(false);
	ui->menuRectangle->setEnabled(false);

	ui->plot->setGridInterval(5);
	on_checkGrid_clicked();

	ui->plot->setRightClickLineCallback([=](QPoint pos) { this->ui->menuLine->popup(pos); });
	ui->plot->setRightClickRectCallback([=](QPoint pos) { this->ui->menuRectangle->popup(pos); });

	m_filterChain.setProcessedDataCallback([this](ndim::pointer<const float, 2> data) {
		ndim::sizes<2> sizes = data.sizes;
		this->ui->spinRoiLeft->setMaximum(sizes[0]);
		this->ui->spinRoiRight->setMaximum(sizes[0]);
		this->ui->spinRoiTop->setMaximum(sizes[1]);
		this->ui->spinRoiBottom->setMaximum(sizes[1]);
	});
	m_filterChain.setPixmapCallback([this](const QPixmap &pixmap) { this->ui->plot->setPixmap(pixmap); });
	m_filterChain.setStatisticCallback([this](const ndimdata::DataStatistic &statistic) {
		this->ui->range->setStatistic(QSharedPointer<ndimdata::DataStatistic>(new ndimdata::DataStatistic(statistic)));
	});

	m_filterChain.setTestPixmapCallback([this](const QPixmap &pixmap) { this->ui->labelTest->setPixmap(pixmap); });

	QVector<QPointF> points;
	points << QPointF(0, 0) << QPointF(1000, 1000);

	QwtPlotCurve *profileCurve = new QwtPlotCurve();
	profileCurve->attach(ui->plotProfile);
	profileCurve->setSamples(points);
	profileCurve->setPen(Qt::black);

	ui->plotProfile->setAutoReplot(true);

	m_filterChain.setProfileCallback([this, profileCurve](ndim::pointer<const float, 1> profile) {
		QVector<QPointF> points;
		int count = int(profile.size());
		points.reserve(count);
		for (int i = 0; i < count; ++i)
			points.append(QPointF(i, profile(i)));
		profileCurve->setSamples(points);
		profileCurve->setVisible(true);
	});

	QwtPlotCurve *zStackCurve = new QwtPlotCurve();
	zStackCurve->attach(ui->plotZStack);
	zStackCurve->setSamples(points);
	zStackCurve->setPen(Qt::black);

	ui->plotZStack->setAutoReplot(true);

	m_filterChain.setZStackCallback([this, zStackCurve](ndim::pointer<const float, 1> zStack) {
		QVector<QPointF> points;
		int count = int(zStack.size());
		points.reserve(count);
		for (int i = 0; i < count; ++i)
			points.append(QPointF(i, zStack(i)));
		zStackCurve->setSamples(points);
		zStackCurve->setVisible(true);
	});

	QTimer *progressTimer = new QTimer(this);
	progressTimer->setSingleShot(false);
	progressTimer->setInterval(100);
	assert_result(connect(&m_filterChain, SIGNAL(validationStarted()), progressTimer, SLOT(start())));
	assert_result(connect(&m_filterChain, SIGNAL(validationDone()), progressTimer, SLOT(stop())));
	assert_result(connect(progressTimer, SIGNAL(timeout()), this, SLOT(updateProgress())));

	assert_result(connect(&m_filterChain, SIGNAL(validationStarted()), this, SLOT(updateProgress())));
	assert_result(connect(&m_filterChain, SIGNAL(validationDone()), this, SLOT(updateProgress())));

	ui->plot->setValueLookup([this](size_t x, size_t y) {
		if (this->m_filterChain.isProcessedDataValid())
			return double(this->m_filterChain.processedData()(x, y));
		else
			return std::nan("");
	});

	QMap<QString, QString> firstMap;
	QMap<QString, QStringList> map;
	int count = 0;

	tw::ProjectLocation project_location(location);
	QStringList files = project_location.getDataFiles();
	for (QString &file : files) {
		file = location.relativeFilePath(file);
		fitshelper::FitsHelper fits(file);
		QMap<QString, QString> keys = fits.readUserKeys();
		if (count == 0)
			firstMap = keys;
		else
			for (auto item = keys.cbegin(), end = keys.cend(); item != end; ++item) {
				if (map.contains(item.key()))
					map[item.key()].append(item.value());
				else if (firstMap.value(item.key()) != item.value()) {
					QStringList list;
					for (int i = 0; i < count; ++i)
						list << firstMap.value(item.key());
					list << item.value();
					map.insert(item.key(), list);
				}
			}
		++count;
	}

	m_keywords = map;
	m_filenames = files;

	ui->comboKeyword->addItem("filename");
	QStringList keywords(map.keys());
	ui->comboKeyword->addItems(keywords);

	ui->listSourceFiles->addItems(files);
	ui->listZStackFiles->addItems(files);
	if (files.size() > 0) {
		ui->listSourceFiles->setCurrentRow(0);
		ui->listZStackFiles->setCurrentRow(0);
	}
	m_filterChain.setDarkImageFilenames(project_location.getDarkImageFiles());
	m_filterChain.setOpenBeamFilenames(project_location.getOpenBeamFiles());

	m_filterChain.start();
}

LiveWidget::~LiveWidget()
{
	delete ui;
}

ndim::range<2> LiveWidget::roi() const
{
	ndim::range<2> range;
	auto horizontal = std::minmax(ui->spinRoiLeft->value(), ui->spinRoiRight->value());
	auto vertical = std::minmax(ui->spinRoiTop->value(), ui->spinRoiBottom->value());
	range.coords[0] = horizontal.first;
	range.coords[1] = vertical.first;
	range.sizes[0] = horizontal.second - horizontal.first;
	range.sizes[1] = vertical.second - vertical.first;
	return range;
}

bool LiveWidget::hasRoi() const
{
	return ui->dockRoi->isVisible();
}

void LiveWidget::resetRoi()
{
	ui->dockRoi->setVisible(false);
	ui->buttonRoiSync->setChecked(false);
	m_filterChain.resetRegionOfInterest();
}

void LiveWidget::setRoi(ndim::range<2> roi)
{
	ui->dockRoi->setVisible(true);
	m_filterChain.setRegionOfInterest(roi);
	ui->spinRoiLeft->setValue(roi.coords[0]);
	ui->spinRoiTop->setValue(roi.coords[1] + roi.sizes[1]);
	ui->spinRoiRight->setValue(roi.coords[0] + roi.sizes[0]);
	ui->spinRoiBottom->setValue(roi.coords[1]);
	ui->lineRoiWidth->setText(QString::number(roi.sizes[0]));
	ui->lineRoiHeight->setText(QString::number(roi.sizes[1]));
}

void LiveWidget::updateProgress()
{
	size_t progress, duration;
	size_t step, stepCount;
	QString description;
	m_filterChain.getStatus(progress, duration, step, stepCount, description);
	if (stepCount)
		description = QString("(%1/%2) %3").arg(step + 1).arg(stepCount).arg(description);
	m_progressLabel->setText(description);
	m_progressBar->setVisible(duration != 0);
	m_progressBar->setValue(int(double(progress) / duration * m_progressBar->maximum()));
	m_progressLabel->setToolTip(m_filterChain.getDescriptions().join('\n'));
}

void LiveWidget::setValidationParameter()
{
	m_filterChain.setImageFilename(m_filenames.value(ui->listSourceFiles->currentIndex().row()));
	LiveWidgetChain::DataSource sourceSelection =
		ui->radioSourceDarkImage->isChecked() ? LiveWidgetChain::DataSource::DarkImage : ui->radioSourceOpenBeam->isChecked() ?
												LiveWidgetChain::DataSource::OpenBeam :
												LiveWidgetChain::DataSource::Image;
	m_filterChain.setDataSource(sourceSelection);
	m_filterChain.setNormalize(ui->checkNormalize->isChecked());
	m_filterChain.setInvert(ui->checkInvert->isChecked());
	m_filterChain.setUseLog(ui->checkLogarithmic->isChecked());
	m_filterChain.setUseColor(ui->checkColor->isChecked());

	switch (ui->range->rangeMode()) {
	case ipw::RangeSelectWidget::RangeManual:
		m_filterChain.setColorRange(ui->range->bounds());
		break;
	case ipw::RangeSelectWidget::RangeFull:
		m_filterChain.setAutoColorRange(true);
		break;
	case ipw::RangeSelectWidget::RangeAuto:
	default:
		m_filterChain.setAutoColorRange(false);
		break;
	}
}

void LiveWidget::on_comboKeyword_currentTextChanged(const QString &keyword)
{
	QStringList list = m_filenames;
	if (ui->listSourceFiles->count() != list.size())
		return;
	if (ui->comboKeyword->currentIndex() != 0)
		list = m_keywords.value(keyword, list);
	for (int i = 0, count = list.size(); i < count; ++i) {
		QString text = list[i];
		if (text.length() > 1 && text[0] == '\'' && text[text.length() - 1] == '\'')
			text = text.midRef(1, text.length() - 2).trimmed().toString();
		ui->listSourceFiles->item(i)->setText(text);
		ui->listZStackFiles->item(i)->setText(text);
	}
}

void LiveWidget::setSelectionMode(ipw::ImagePlot::SelectionMode mode)
{
	ui->actionZoom->setChecked(mode == ipw::ImagePlot::SelectionZoom);
	ui->actionLine->setChecked(mode == ipw::ImagePlot::SelectionLine);
	ui->actionRectangle->setChecked(mode == ipw::ImagePlot::SelectionRect);
	ui->plot->setSelectionMode(mode);
	ui->plot->resetSelection();
	ui->menuLine->setEnabled(false);
	ui->menuRectangle->setEnabled(false);
}

void LiveWidget::on_actionZoom_triggered()
{
	setSelectionMode(ipw::ImagePlot::SelectionZoom);
}

void LiveWidget::on_actionLine_triggered()
{
	setSelectionMode(ipw::ImagePlot::SelectionLine);
}

void LiveWidget::on_actionRectangle_triggered()
{
	setSelectionMode(ipw::ImagePlot::SelectionRect);
}

void LiveWidget::on_plot_selectionChanged(QRectF)
{
	ui->menuRectangle->setEnabled(true);
	if (ui->dockRoi->isVisible() && ui->buttonRoiSync->isChecked()) {
		m_updating = true;
		on_actionSetRoi_triggered();
		m_updating = false;
	}
}

void LiveWidget::on_plot_selectionChanged(QLineF)
{
	ui->menuLine->setEnabled(true);
	m_filterChain.setProfileLine(ui->plot->selectionLine());
}

void LiveWidget::on_actionResetView_triggered()
{
	QRectF rect;
	switch (ui->plot->selectionMode()) {
	case ipw::ImagePlot::SelectionLine: {
		QLineF line = ui->plot->selectionLine();
		rect = QRectF(line.p1(), line.p2()).normalized();
		break;
	}
	case ipw::ImagePlot::SelectionRect:
		rect = ui->plot->selectionRect().normalized();
		break;
	default: {
		QPixmap pixmap = ui->plot->pixmap();
		ui->plot->setView(QRectF(QPointF(0, 0), QSizeF(pixmap.size())));
		return;
	}
	}
	QSizeF adjust = rect.size() * .05;
	rect.adjust(-adjust.width(), -adjust.height(), adjust.width(), adjust.height());
	ui->plot->setView(rect);
}

void LiveWidget::on_checkGrid_clicked()
{
	ui->plot->setGridMode(ui->checkGrid->isChecked() ? ipw::ImagePlot::GridCustom : ipw::ImagePlot::GridDisable);
}

void LiveWidget::on_actionResetSelection_triggered()
{
	ui->plot->resetSelection();
	ui->menuLine->setEnabled(false);
	ui->menuRectangle->setEnabled(false);
	ui->dockProfile->setVisible(false);
}

void LiveWidget::on_actionPlotProfile_triggered()
{
	ui->dockProfile->setVisible(true);
	m_filterChain.enableProfile();
}

void LiveWidget::on_actionSetRoi_triggered()
{
	QRectF rectF = ui->plot->selectionRect();
	QRect rect = rectF.toRect();
	if (!m_filterChain.isProcessedDataValid())
		return; // TODO: Was, wenn sich Daten ändern...
	ndim::sizes<2> sizes = m_filterChain.processedData().sizes;
	rect = rect.intersected(QRect(0, 0, sizes[0], sizes[1]));
	ndim::range<2> region;
	region.coords[0] = rect.left();
	region.coords[1] = rect.top();
	region.sizes[0] = rect.width();
	region.sizes[1] = rect.height();
	m_updating = true;
	setRoi(region);
	ui->buttonRoiSync->setChecked(true);
	m_updating = false;
}

void LiveWidget::on_actionZStackPlot_triggered()
{
	ui->dockZStackPlot->setVisible(true);
	ui->dockZStackSource->setVisible(true);

	QRectF rectF = ui->plot->selectionRect();
	QRect rect = rectF.toRect();
	if (!m_filterChain.isProcessedDataValid())
		return; // TODO: Was, wenn sich Daten ändern...
	ndim::sizes<2> sizes = m_filterChain.processedData().sizes;
	rect = rect.intersected(QRect(0, 0, sizes[0], sizes[1]));
	ndim::range<2> region;
	region.coords[0] = rect.left();
	region.coords[1] = rect.top();
	region.sizes[0] = rect.width();
	region.sizes[1] = rect.height();

	m_filterChain.setZStackRegion(region);

	m_filterChain.enableZStack();
}

void LiveWidget::on_listZStackFiles_itemSelectionChanged()
{
	auto items = ui->listZStackFiles->selectionModel()->selectedIndexes();
	QStringList filenames;
	std::transform(items.cbegin(), items.cend(), std::back_inserter(filenames),
		[this](const QModelIndex &index) { return this->m_filenames.value(index.row()); });
	m_filterChain.setZStackFiles(filenames);
}

void LiveWidget::on_listSourceFiles_currentItemChanged(QListWidgetItem *, QListWidgetItem *)
{
	setValidationParameter();
}

void LiveWidget::on_radioSourceImage_clicked()
{
	setValidationParameter();
}

void LiveWidget::on_radioSourceOpenBeam_clicked()
{
	setValidationParameter();
}

void LiveWidget::on_radioSourceDarkImage_clicked()
{
	setValidationParameter();
}

void LiveWidget::on_checkColor_clicked()
{
	setValidationParameter();
}

void LiveWidget::on_checkInvert_clicked()
{
	setValidationParameter();
}

void LiveWidget::on_checkNormalize_clicked()
{
	setValidationParameter();
}

void LiveWidget::on_checkLogarithmic_clicked()
{
	setValidationParameter();
}

void LiveWidget::on_range_rangeChanged(double, double)
{
	setValidationParameter();
}

void LiveWidget::on_buttonRoiSync_clicked(bool checked)
{
	if (!checked || m_updating)
		return;
	ui->plot->setSelectionMode(ipw::ImagePlot::SelectionRect);
	ndim::range<2> roi = this->roi();
	QPointF point(float(roi.coords[0]), float(roi.coords[1]));
	QSizeF size(float(roi.sizes[0]), float(roi.sizes[1]));
	m_updating = true;
	ui->plot->setSelection(QRectF(point, size));
	m_updating = false;
}

void LiveWidget::on_buttonRoiReset_clicked()
{
	resetRoi();
}

} // namespace lw
