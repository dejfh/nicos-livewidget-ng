#ifndef PYFC_VALIDATOR_H
#define PYFC_VALIDATOR_H

#include <QObject>

#include <boost/lockfree/spsc_queue.hpp>

#include "fc/validation/validator.h"

namespace pyfc
{

class Validator : public QObject, protected fc::validation::Validator
{
	Q_OBJECT

public:
	explicit Validator(QObject *parent = 0);

signals:
	void dispatched();

private slots:
	void doNow();
};

} // namespace pyfc

#endif // PYFC_VALIDATOR_H
