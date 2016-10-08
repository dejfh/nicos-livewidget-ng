class QtValidator : public QObject
{
%TypeHeaderCode
#include "fc/validation/qtvalidator.h"
#include "pyfc/pyfilter.h"
using QtValidator = fc::validation::QtValidator;
%End

public:
	explicit QtValidator(QObject *parent = 0);

signals:
	void procDone();
	void validationStarted();
	void validationStep();
	void validationComplete();
	void invalidated();

public:

	bool isWorking() const;

	void start();
	void abort(bool wait);

	void restart(bool wait);
	void add(Validatable validatable);
	void remove(Validatable validatable);
	%MethodCode
		sipCpp->remove(a0->get());
	%End

	// void state(size_t &progress, size_t &duration, size_t &step, size_t &stepCount, QString &description);
	QStringList descriptions() const;
};
