#ifndef HELPER_QT_DISPATCHER_H
#define HELPER_QT_DISPATCHER_H

#include <deque>

#include <QObject>

#include "helper/dispatcher.h"
#include "helper/threadsafe.h"

namespace hlp
{
namespace qt
{

class Dispatcher : public QObject, public virtual hlp::Dispatcher
{
	Q_OBJECT

	// Dispatcher interface
public:
	Dispatcher()
	{
		connect(this, SIGNAL(newItem()), this, SLOT(invokeNow()));
	}

	virtual void invoke(std::function<void(void)> operation) override
	{
		auto guard = m_queue.lock();
		guard.data().push_back(std::move(operation));
		emit newItem();
	}

signals:
	void newItem();

private slots:
	void invokeNow()
	{
		for (;;) {
			std::function<void(void)> operation;
			{
				auto guard = m_queue.lock();
				auto &queue = guard.data();
				if (queue.empty())
					return;
				operation = std::move(queue.front());
				queue.pop_front();
			}
			operation();
		}
	}

private:
	hlp::Threadsafe<std::deque<std::function<void(void)>>> m_queue;
};

} // namespace qt
} // namespace hlp

#endif // HELPER_QT_DISPATCHER_H
