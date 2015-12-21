#ifndef FITS_FITSLOADERSETTINGS_H
#define FITS_FITSLOADERSETTINGS_H

#include <QString>
#include <QStringList>

class QSettings;

struct FitsLoaderSettings {
    enum initialization { default_settings };

    static const FitsLoaderSettings &defaults();

    QStringList angle_fits_keys;
    QStringList angle_file_patterns;
    int mean_drop_low;
    int mean_drop_high;

    FitsLoaderSettings() {}
    explicit FitsLoaderSettings(initialization);
    explicit FitsLoaderSettings(const QSettings &settings, const QString &prefix = QString());

    void save(QSettings &settings, const QString &prefix) const;
};

#endif // FITS_FITSLOADERSETTINGS_H
