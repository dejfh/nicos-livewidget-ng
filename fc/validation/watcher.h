#ifndef FILTER_VALIDATION_WATCHER_H
#define FILTER_VALIDATION_WATCHER_H

#include <functional>

#include <QObject>

#include "fc/validation/validator.h"
#include "fc/filter.h"

namespace fc
{

namespace validation
{

class Watcher : public QObject
{
	Q_OBJECT

	std::shared_ptr<const fc::Validatable> m_validatable;
	bool m_isUpToDate;
	std::function<void(void)> m_updater;

public:
	explicit Watcher(Validator *validator, std::shared_ptr<const fc::Validatable> validatable,
		const std::function<void(void)> &updater = std::function<void(void)>(), QObject *parent = 0);

	void setUpdater(std::function<void(void)> updater);

public slots:
	void validated();
	void invalidated();
};

} // namespace validation

} // namespace fc

#endif // FILTER_VALIDATION_WATCHER_H
