#include <QApplication>
#include <QMetaType>
#include <QSharedPointer>
#include <QScopedPointer>

#include "initdialog.h"

#include "lw/livewidget.h"

#include <memory>

#include "ndim/layout.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	QDir location;

	{
		InitDialog dialog;
		dialog.show();
		int r = a.exec();
		if (!r)
			return 1;
		location = dialog.location();
	}

	QScopedPointer<lw::LiveWidget> liveWidget(new lw::LiveWidget(location));
	liveWidget->show();
	return a.exec();
}
