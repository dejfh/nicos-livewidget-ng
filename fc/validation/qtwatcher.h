#ifndef FILTER_VALIDATION_WATCHER_H
#define FILTER_VALIDATION_WATCHER_H

#include <functional>

#include <QObject>

#include "fc/validation/qtvalidator.h"
#include "fc/filter.h"
#include "helper/assertcast.h"

namespace fc
{

namespace validation
{

class QtWatcher : public QObject
{
	Q_OBJECT

	std::shared_ptr<const fc::Validatable> m_validatable;
	bool m_isUpToDate;

public:
	explicit QtWatcher(QtValidator *validator, std::shared_ptr<const fc::Validatable> validatable, QObject *parent = 0)
		: QObject(parent)
		, m_validatable(std::move(validatable))
		, m_isUpToDate(false)
	{
		connect(validator, SIGNAL(validationStep()), this, SLOT(on_validated()));
		connect(validator, SIGNAL(invalidated()), this, SLOT(on_invalidated()));
	}

signals:
	void validated();
	void invalidated();

private slots:
	void on_validated()
	{
		if (!m_isUpToDate && m_validatable->isValid()) {
			m_isUpToDate = true;
			emit validated();
		}
	}

	void on_invalidated()
	{
		if (m_isUpToDate && !m_validatable->isValid()) {
			m_isUpToDate = false;
			emit invalidated();
		}
	}
};

} // namespace validation

} // namespace fc

#endif // FILTER_VALIDATION_WATCHER_H
