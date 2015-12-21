#ifndef HELPER_DISPATCHER_H
#define HELPER_DISPATCHER_H

#include <functional>

namespace hlp
{

class Dispatcher
{
public:
	virtual void invoke(std::function<void(void)> operation) = 0;
};

} // namespace hlp

#endif // HELPER_DISPATCHER_H
