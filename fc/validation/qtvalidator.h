#ifndef FC_VALIDATION_QTVALIDATOR_H
#define FC_VALIDATION_QTVALIDATOR_H

#include <QObject>

#include "fc/validation/validator.h"

namespace fc
{
namespace validation
{

class QtValidator : public QObject, public Validator
{
	Q_OBJECT

public:
	explicit QtValidator(QObject *parent = 0);

signals:
	void procDone();
	void validationStarted();
	void validationStep();
	void validationComplete();
	void invalidated();

public slots:

private slots:
	void syncInvoke();

	// Validator interface
protected:
	virtual void invokeFinished() override;
	virtual void onValidationStarted() override;
	virtual void onValidationStep() override;
	virtual void onValidationComplete() override;
	virtual void onInvalidated() override;
};

} // namespace validation
} // namespace fc

#endif // FC_VALIDATION_QTVALIDATOR_H
