#ifndef TW_PROJECTLOCATIONSETTINGS_H
#define TW_PROJECTLOCATIONSETTINGS_H

#include "fits/fitsloadersettings.h"

#include <QString>
#include <QStringList>

class QSettings;

namespace tw
{

struct ProjectLocationSettings : FitsLoaderSettings {
	enum initialization { default_settings };

	QStringList darkimage_folders;
	QStringList darkimage_patterns;
	QStringList openbeam_folders;
	QStringList openbeam_patterns;
	QStringList raw_folders;
	QStringList raw_patterns;

	ProjectLocationSettings()
	{
	}
	explicit ProjectLocationSettings(initialization);
	explicit ProjectLocationSettings(const QSettings &settings, const QString &prefix = QString());

	void save(QSettings &settings, const QString &prefix) const;
};

} // namespace tw
#endif // TW_PROJECTLOCATIONSETTINGS_H
