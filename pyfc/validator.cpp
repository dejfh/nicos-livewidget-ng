#include "validator.h"

#include <helper/helper.h>

namespace pyfc
{

Validator::Validator(QObject *parent)
	: QObject(parent)
{
	hlp::assert_result(connect(this, SIGNAL(dispatched()), this, SLOT(doNow()), Qt::QueuedConnection));
}

void Validator::doNow()
{
	auto consumer = [](const std::function<void()> &func) { func(); };
	m_dispatchQueue.consume_all(consumer);
}

void Validator::dispatch(std::function<void()> op)
{
}

} // namespace pyfc
