#include "qtvalidator.h"

#include "helper/helper.h"

using hlp::assert_true;

namespace fc
{
namespace validation
{

QtValidator::QtValidator(QObject *parent)
	: QObject(parent)
{
	assert_true() << connect(this, SIGNAL(procDone()), this, SLOT(syncInvoke()), Qt::QueuedConnection);
}

void QtValidator::syncInvoke()
{
	this->finished();
}

void QtValidator::invokeFinished()
{
	emit procDone();
}

void QtValidator::onValidationStarted()
{
	emit validationStarted();
}

void QtValidator::onValidationStep()
{
	emit validationStep();
}

void QtValidator::onValidationComplete()
{
	emit validationComplete();
}

void QtValidator::onInvalidated()
{
	emit invalidated();
}

} // namespace validation
} // namespace fc
