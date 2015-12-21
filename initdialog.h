#ifndef TS_INITDIALOG_H
#define TS_INITDIALOG_H

#include <QWidget>

#include <QDir>
#include <QString>

#include <QScopedPointer>

class QThread;
class SignalingProgress;
namespace tw
{
class ProjectLocation;
} // namespace tw
namespace tomo
{
class SinogramFile;
} // namespace tomo

class QSettings;

struct InitDialogSettings;

namespace Ui
{
class InitDialog;
}

class InitDialog : public QWidget
{
	Q_OBJECT

	QScopedPointer<InitDialogSettings> m_settings;

	QScopedPointer<tw::ProjectLocation> projectLocation;

public:
	explicit InitDialog(QWidget *parent = 0);
	~InitDialog();

	void findDatasets();

	const QDir &location() const;

private slots:
	void on_buttonDataPath_clicked();

	void on_buttonOk_clicked();
	void on_buttonCancel_clicked();

private:
	Ui::InitDialog *ui;
};

#endif // TS_INITDIALOG_H
