#ifndef FC_VALIDATION_QTWATCHER_H
#define FC_VALIDATION_QTWATCHER_H

#include <QObject>

namespace fc {
namespace validation {

class QtWatcher : public QObject
{
	Q_OBJECT
public:
	explicit QtWatcher(QObject *parent = 0);

signals:

public slots:
};

} // namespace validation
} // namespace fc

#endif // FC_VALIDATION_QTWATCHER_H
