#ifndef TOMO_RECONSTRUCTORERRORS_H
#define TOMO_RECONSTRUCTORERRORS_H

#include <stdexcept>

namespace tomo
{

class not_available_error : public std::logic_error
{
public:
    not_available_error()
        : std::logic_error(std::string("There is no completed request available."))
    {
    }
};

} // namespace tomo

#endif // TOMO_RECONSTRUCTORERRORS_H
