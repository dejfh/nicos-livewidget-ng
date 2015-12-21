#ifndef TS_INITDIALOGSETTINGS_H
#define TS_INITDIALOGSETTINGS_H

#include <QString>
#include <QStringList>

class QSettings;

struct InitDialogSettings {
    enum initialization { default_settings };

    QString project_location;

    InitDialogSettings() {}
    explicit InitDialogSettings(initialization);
    explicit InitDialogSettings(const QSettings &settings, const QString &prefix = QString());

    void save(QSettings &settings, const QString &prefix = QString()) const;
};

#endif // TS_INITDIALOGSETTINGS_H
