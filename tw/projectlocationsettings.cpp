#include "tw/projectlocationsettings.h"

#include <QSettings>

namespace tw
{

const char key_darkiamge_folders[] = "darkimage_folders";
const char key_darkiamge_patterns[] = "darkimage_patterns";
const char key_openbeam_folders[] = "openbeam_folders";
const char key_openbeam_patterns[] = "openbeam_patterns";
const char key_raw_folders[] = "raw_folders";
const char key_raw_patterns[] = "raw_patterns";

const ProjectLocationSettings _default_settings = ProjectLocationSettings(ProjectLocationSettings::default_settings);

ProjectLocationSettings::ProjectLocationSettings(ProjectLocationSettings::initialization)
	: FitsLoaderSettings(FitsLoaderSettings::default_settings)
{
	darkimage_folders << "di"
					  << "darkimage";
	darkimage_patterns << "di_*.fits"
					   << "*.fits";

	openbeam_folders << "ob"
					 << "openbeam";
	openbeam_patterns << "ob_*.fits"
					  << "*.fits";

	raw_folders << "raw"
				<< ".";
	raw_patterns << "*_???.???.fits"
				 << "*.fits";
}

ProjectLocationSettings::ProjectLocationSettings(const QSettings &settings, const QString &prefix)
	: FitsLoaderSettings(settings, prefix)
{
	darkimage_folders = settings.value(prefix + key_darkiamge_folders, _default_settings.darkimage_folders).toStringList();
	darkimage_patterns = settings.value(prefix + key_darkiamge_patterns, _default_settings.darkimage_patterns).toStringList();
	openbeam_folders = settings.value(prefix + key_openbeam_folders, _default_settings.openbeam_folders).toStringList();
	openbeam_patterns = settings.value(prefix + key_openbeam_patterns, _default_settings.openbeam_patterns).toStringList();
	raw_folders = settings.value(prefix + key_raw_folders, _default_settings.raw_folders).toStringList();
	raw_patterns = settings.value(prefix + key_raw_patterns, _default_settings.raw_patterns).toStringList();
}

void ProjectLocationSettings::save(QSettings &settings, const QString &prefix) const
{
	FitsLoaderSettings::save(settings, prefix);
	settings.setValue(prefix + key_darkiamge_folders, darkimage_folders);
	settings.setValue(prefix + key_darkiamge_patterns, darkimage_patterns);
	settings.setValue(prefix + key_openbeam_folders, openbeam_folders);
	settings.setValue(prefix + key_openbeam_patterns, openbeam_patterns);
	settings.setValue(prefix + key_raw_folders, raw_folders);
	settings.setValue(prefix + key_raw_patterns, raw_patterns);
}

} // namespace tw
