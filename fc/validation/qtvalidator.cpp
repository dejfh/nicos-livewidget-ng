#include "qtvalidator.h"

namespace fc
{
namespace validation
{

QtValidator::QtValidator(QObject *parent)
	: QObject(parent)
{
	hlp::assert_true() << connect(this, SIGNAL(procDone()), this, SLOT(invokeFinishedSync()), Qt::QueuedConnection);
}

void QtValidator::invokeFinishedSync()
{
	this->finished();
}

void QtValidator::invokeFinished()
{
	emit procDone();
}

void QtValidator::onPrepareProc()
{
	emit prepareProcCalled();
}

void QtValidator::onValidationProc()
{
	emit validationProcCalled();
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
