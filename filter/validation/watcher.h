#ifndef FILTER_VALIDATION_WATCHER_H
#define FILTER_VALIDATION_WATCHER_H

#include <functional>

#include <QObject>

#include "filter/validation/validator.h"
#include "filter/filter.h"

namespace filter
{

namespace validation
{

class Watcher : public QObject
{
	Q_OBJECT

	std::shared_ptr<const filter::Validatable> m_validatable;
	bool m_isUpToDate;
	std::function<void(void)> m_updater;

public:
	explicit Watcher(Validator *validator, std::shared_ptr<const filter::Validatable> validatable,
		const std::function<void(void)> &updater = std::function<void(void)>(), QObject *parent = 0);

	void setUpdater(std::function<void(void)> updater);

public slots:
	void validated();
	void invalidated();
};

} // namespace validation

} // namespace filter

#endif // FILTER_VALIDATION_WATCHER_H
