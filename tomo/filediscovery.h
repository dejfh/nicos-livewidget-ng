#ifndef TOMO_FILEDISCOVERY_H
#define TOMO_FILEDISCOVERY_H

#include <QObject>

#include <QFileSystemWatcher>

namespace tomo {

class FileDiscovery : public QObject
{
	Q_OBJECT

	QFileSystemWatcher m_watcher;
public:
	explicit FileDiscovery(QObject *parent = 0);

signals:

public slots:
};

} // namespace tomo

#endif // TOMO_FILEDISCOVERY_H
