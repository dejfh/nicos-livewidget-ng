#include "tw/projectlocation.h"

#include "tw/projectlocationsettings.h"

#include "fits/fitsloader.h"

#include <QDir>

namespace tw
{

ProjectLocation::ProjectLocation(const QDir &location)
	: m_settings(new ProjectLocationSettings(ProjectLocationSettings::default_settings))
	, m_location(location)
{
}

ProjectLocation::~ProjectLocation()
{
}

const QDir &ProjectLocation::location() const
{
	return m_location;
}

void ProjectLocation::setSettings(const ProjectLocationSettings &settings)
{
	*m_settings = settings;
}
const ProjectLocationSettings &ProjectLocation::settings() const
{
	return *m_settings;
}

void ProjectLocation::setSettings(const QSettings &settings, const QString &prefix)
{
	*m_settings = ProjectLocationSettings(settings, prefix);
}
void ProjectLocation::getSettings(QSettings &settings, const QString &prefix) const
{
	m_settings->save(settings, prefix);
}

QStringList get_files(const QDir &parent, const QStringList &folders, const QStringList &name_filters)
{
	for (const QString &folder : folders) {
		QDir dir(parent.absoluteFilePath(folder));
		QStringList nameFilter;
		nameFilter << QString();
		if (dir.exists()) {
			for (const QString &pattern : name_filters) {
				nameFilter.front() = pattern;
				QStringList files = dir.entryList(nameFilter, QDir::Files);
				if (files.count() > 0) {
					for (QString &file : files)
						file = dir.absoluteFilePath(file);
					return files;
				}
			}
		}
	}
	return QStringList();
}

QStringList ProjectLocation::getDarkImageFiles() const
{
	return get_files(m_location, m_settings->darkimage_folders, m_settings->darkimage_patterns);
}

QStringList ProjectLocation::getOpenBeamFiles() const
{
	return get_files(m_location, m_settings->openbeam_folders, m_settings->openbeam_patterns);
}

QStringList ProjectLocation::getDataFiles() const
{
	return get_files(m_location, m_settings->raw_folders, m_settings->raw_patterns);
}

// angles in degrees
QMap<QString, float> ProjectLocation::getDataFilesWithAngles(AsyncFloatProgress progress) const
{
	const QStringList filenames = getDataFiles();
	return getDataFilesWithAngles(progress, filenames);
}

// angles in degrees
QMap<QString, float> ProjectLocation::getDataFilesWithAngles(AsyncFloatProgress progress, QStringList filenames) const
{
	double progress_step = 1.0 / filenames.count();
	QMap<QString, float> map;
	for (const QString &filename : filenames) {
		fitshelper::TomoFitsHelper fits(filename);
		double angle = fits.getAngle(*m_settings);
		if (progress.step(progress_step))
			return map;
		if (std::isnan(angle))
			continue;
		map.insert(filename, angle);
	}
	return map;
}

} // namespace tw
