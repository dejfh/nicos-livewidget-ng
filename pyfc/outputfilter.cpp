#include "outputfilter.h"

#include "helper/helper.h"

using hlp::assert_true;

namespace pyfc
{

OutputFilterBase::OutputFilterBase(fc::validation::QtValidator *validator)
	: QObject(validator)
	, m_updated(false)
{
	assert_true() << connect(validator, SIGNAL(invalidated()), this, SLOT(checkInvalid()));
	assert_true() << connect(validator, SIGNAL(validationStep()), this, SLOT(checkValid()));
}

fc::validation::QtValidator *OutputFilterBase::validator()
{
	return m_validator;
}

void OutputFilterBase::checkInvalid()
{
	if (m_updated && !this->ownValidatable().isValid()) {
		m_updated = false;
		emit invalidated();
	}
}

void OutputFilterBase::checkValid()
{
	if (!m_updated && this->ownValidatable().isValid()) {
		m_updated = true;
		this->emitValidated();
	}
}

} // namespace pyfc
