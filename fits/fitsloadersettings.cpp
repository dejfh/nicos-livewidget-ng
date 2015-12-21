#include "fits/fitsloadersettings.h"

#include <QSettings>

const char key_angle_fits_keys[] = "angle_fits_keys";
const char key_angle_file_patterns[] = "angle_file_patterns";
const char key_mean_drop_low[] = "mean_drop_low";
const char key_mean_drop_high[] = "mean_drop_high";

const FitsLoaderSettings _default_settings = FitsLoaderSettings(FitsLoaderSettings::default_settings);

const FitsLoaderSettings &FitsLoaderSettings::defaults()
{
	return _default_settings;
}

FitsLoaderSettings::FitsLoaderSettings(FitsLoaderSettings::initialization)
{
	angle_fits_keys << "sry/value";
	angle_file_patterns << "_(\\d\\d\\d\\.\\d\\d\\d)\\.fits$";
	mean_drop_low = 2;
	mean_drop_high = 2;
}

FitsLoaderSettings::FitsLoaderSettings(const QSettings &settings, const QString &prefix)
{
	angle_fits_keys = settings.value(prefix + key_angle_fits_keys, _default_settings.angle_fits_keys).toStringList();
	angle_file_patterns = settings.value(prefix + key_angle_file_patterns, _default_settings.angle_file_patterns).toStringList();
	mean_drop_low = settings.value(prefix + key_mean_drop_low, _default_settings.mean_drop_low).toInt();
	mean_drop_high = settings.value(prefix + key_mean_drop_high, _default_settings.mean_drop_high).toInt();
}

void FitsLoaderSettings::save(QSettings &settings, const QString &prefix) const
{
	settings.setValue(prefix + key_angle_fits_keys, angle_fits_keys);
	settings.setValue(prefix + key_angle_file_patterns, angle_file_patterns);
	settings.setValue(prefix + key_mean_drop_low, mean_drop_low);
	settings.setValue(prefix + key_mean_drop_high, mean_drop_high);
}
