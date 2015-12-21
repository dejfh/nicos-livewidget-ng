#include "initdialogsettings.h"

#include <QSettings>

static const char key_project_location[] = "project_location";

InitDialogSettings _default_settings = InitDialogSettings(InitDialogSettings::default_settings);

InitDialogSettings::InitDialogSettings(InitDialogSettings::initialization)
{ //
    project_location = ".";
}

InitDialogSettings::InitDialogSettings(const QSettings &settings, const QString &prefix)
{ //
    project_location = settings.value(prefix + key_project_location, _default_settings.project_location).toString();
}

void InitDialogSettings::save(QSettings &settings, const QString &prefix) const
{ //
    settings.setValue(prefix + key_project_location, project_location);
}
