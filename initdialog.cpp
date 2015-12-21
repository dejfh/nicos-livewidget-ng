#include "initdialog.h"
#include "ui_initdialog.h"

#include <QFileDialog>
#include <QDir>
#include <QFileInfo>

#include <QApplication>

#include <QButtonGroup>

#include <QMultiMap>

#include <QSettings>

#include "initdialogsettings.h"
#include "tw/projectlocation.h"

InitDialog::InitDialog(QWidget *parent)
	: QWidget(parent)
	, m_settings(0)
	, ui(new Ui::InitDialog)
{
	QSettings settings("config.ini", QSettings::IniFormat);
	//    QSettings settings(QSettings::IniFormat, QSettings::UserScope, "FRM2", "Heatmap Widget", this);
	m_settings.reset(new InitDialogSettings(settings));
	m_settings->save(settings);

	ui->setupUi(this);

	findDatasets();
}

InitDialog::~InitDialog()
{
	delete ui;
}

void InitDialog::findDatasets()
{
	QDir dir(m_settings->project_location);
	QStringList dirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
	for (const QString &dirName : dirs) {
		QString path = QDir::toNativeSeparators(dir.absoluteFilePath(dirName));
		ui->cobDataPath->addItem(dirName, path);
	}
}

const QDir &InitDialog::location() const
{
	return projectLocation->location();
}

QString getComboItem(QComboBox &control)
{
	int index = control.currentIndex();
	QString text = control.currentText();
	if (index < 0 || text != control.itemText(index))
		return text;
	else {
		QString data = control.currentData().toString();
		return data.isEmpty() ? text : data;
	}
}

void showDirSearchDialog(QComboBox &control)
{
	QString dirName = QFileDialog::getExistingDirectory(0, QString(), QDir(getComboItem(control)).absoluteFilePath(".."));
	if (!dirName.isEmpty())
		control.setCurrentText(QDir::toNativeSeparators(dirName));
}

void showFileSearchDialog(QComboBox &control, const QString &fileExtension)
{
	QFileInfo fileInfo(getComboItem(control));
	QString fileName = QFileDialog::getSaveFileName(0, QString(), fileInfo.dir().absolutePath(), fileExtension);
	if (!fileName.isEmpty())
		control.setCurrentText(QDir::toNativeSeparators(fileName));
}

void InitDialog::on_buttonDataPath_clicked()
{
	showDirSearchDialog(*ui->cobDataPath);
}

void InitDialog::on_buttonOk_clicked()
{
	projectLocation.reset(new tw::ProjectLocation(getComboItem(*ui->cobDataPath)));

	qApp->exit(1);
	return;
}

void InitDialog::on_buttonCancel_clicked()
{
	close();
}
