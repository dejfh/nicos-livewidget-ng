#ifndef PYFC_OUTPUTFILTER_H
#define PYFC_OUTPUTFILTER_H

#include <memory>

#include <QObject>

#include <fc/datafilter.h>

#include <fc/validation/qtvalidator.h>

namespace pyfc
{
class OutputFilterBase : public QObject
{
	Q_OBJECT

	fc::validation::QtValidator *m_validator;
	bool m_updated;

public:
	explicit OutputFilterBase(fc::validation::QtValidator *validator);

	fc::validation::QtValidator *validator();

protected:
	virtual const fc::Validatable &ownValidatable() const = 0;
	virtual void emitValidated() = 0;

signals:
	void invalidated();

private slots:
	void checkInvalid();
	void checkValid();
};

} // namespace pyfc

#endif // PYFC_OUTPUTFILTER_H
