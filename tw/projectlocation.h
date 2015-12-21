#ifndef TW_PROJECTLOCATION_H
#define TW_PROJECTLOCATION_H

#include <QScopedPointer>
#include <QString>
#include <QDir>
#include <QStringList>
#include <QMap>

#include "asyncprogress.h"

class QSettings;

namespace tw
{

struct ProjectLocationSettings;

class ProjectLocation
{
	QScopedPointer<ProjectLocationSettings> m_settings;
	QDir m_location;

public:
	//    ProjectLocation() {}
	ProjectLocation(const QDir &location);
	~ProjectLocation();

	const QDir &location() const;

	void setSettings(const ProjectLocationSettings &settings);
	const ProjectLocationSettings &settings() const;

	void setSettings(const QSettings &settings, const QString &prefix = QString());
	void getSettings(QSettings &settings, const QString &prefix = QString()) const;

	QStringList getDarkImageFiles() const;
	QStringList getOpenBeamFiles() const;
	QStringList getDataFiles() const;
	QMap<QString, float> getDataFilesWithAngles(AsyncFloatProgress progress) const;
	QMap<QString, float> getDataFilesWithAngles(AsyncFloatProgress progress, QStringList filenames) const;
};

} // namespace tw

#endif // TW_PROJECTLOCATION_H
