#include "fc/validation/watcher.h"

#include "safecast.h"

using jfh::assert_result;

namespace fc
{

namespace validation
{

Watcher::Watcher(
	Validator *validator, std::shared_ptr<const fc::Validatable> validatable, const std::function<void(void)> &updater, QObject *parent)
	: QObject(parent)
	, m_validatable(validatable)
	, m_isUpToDate(false)
	, m_updater(updater)
{
	assert_result(connect(validator, SIGNAL(validationStep()), this, SLOT(validated())));
	assert_result(connect(validator, SIGNAL(invalidated()), this, SLOT(invalidated())));
}

void Watcher::setUpdater(std::function<void()> updater)
{
	m_updater = std::move(updater);
	if (m_isUpToDate && m_updater)
		m_updater();
}

void Watcher::validated()
{
	//	if (m_isUpToDate)
	//		return;
	std::shared_ptr<const fc::Validatable> validatable(m_validatable);
	if (!validatable || !validatable->isValid())
		return;
	if (m_updater)
		m_updater();
	m_isUpToDate = true;
}

void Watcher::invalidated()
{
	//	if (!m_isUpToDate)
	//		return;
	std::shared_ptr<const fc::Validatable> validatable(m_validatable);
	if (!validatable || !validatable->isValid())
		m_isUpToDate = false;
}

} // namespace validation

} // namespace fc
