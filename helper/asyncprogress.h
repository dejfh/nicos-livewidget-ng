#ifndef ASYNCPROGRESS_H
#define ASYNCPROGRESS_H

#include <cstddef>
#include <atomic>
#include <stdexcept>

namespace hlp {

class OperationCanceledException : public std::runtime_error
{
public:
    OperationCanceledException()
        : std::runtime_error("Validation cancelled.")
    {
    }
};

class AsyncTask
{
protected:
    std::atomic<bool> m_cancelled;

public:
    AsyncTask()
        : m_cancelled(false)
    {
    }
    AsyncTask(const AsyncTask &other)
        : m_cancelled(other.m_cancelled.load())
    {
    }
    AsyncTask &operator=(const AsyncTask &other)
    {
        this->m_cancelled = other.m_cancelled.load();
        return *this;
    }

    bool isCancelled() const
    {
        return m_cancelled;
    }

    void throwIfCancelled() const
    {
        if (this->isCancelled())
            throw OperationCanceledException();
    }

    void reset()
    {
        m_cancelled = false;
    }
    void cancel()
    {
        m_cancelled = true;
    }
};

/*! \brief Notifies about cancelation and collects the progress during an async process.
 * */
template <typename _ValueType>
class AsyncProgress : public AsyncTask
{
public:
    using ValueType = _ValueType;

protected:
    std::atomic<ValueType> m_progress;

public:
    AsyncProgress()
        : m_progress(0)
    {
    }
    AsyncProgress(const AsyncProgress &other)
        : AsyncTask(other)
        , m_progress(other.m_progress.load())
    {
    }
    AsyncProgress &operator=(const AsyncProgress &other)
    {
        static_cast<AsyncTask &>(*this) = other;
        m_progress = other.m_progress.load();
        return *this;
    }

    void reset()
    {
        AsyncTask::reset();
        m_progress = 0;
    }
    void cancel()
    {
        AsyncTask::cancel();
    }

    void advanceProgress(ValueType flops)
    {
        flops += m_progress.load(std::memory_order_relaxed);
        m_progress.store(flops, std::memory_order_relaxed);
    }
    void setProgress(ValueType progress)
    {
        m_progress.store(progress, std::memory_order_relaxed);
    }

    ValueType progress() const
    {
        return m_progress;
    }
};

class AsyncPartialProgress
{
protected:
    AsyncProgress<float> *progress;
    float factor;
    float final;

public:
    AsyncPartialProgress(AsyncProgress<float> *progress, float step)
        : progress(progress)
        , factor(step)
        , final(progress ? progress->progress() + step : step)
    {
    }
    ~AsyncPartialProgress()
    {
        if (progress)
            progress->setProgress(final);
    }
    AsyncPartialProgress(const AsyncPartialProgress &other) = delete;

    AsyncPartialProgress(AsyncPartialProgress &&other)
        : progress(other.progress)
        , factor(other.factor)
        , final(other.final)
    {
        other.progress = nullptr;
    }
    AsyncPartialProgress &operator=(AsyncPartialProgress &&other)
    {
        if (progress)
            progress->setProgress(final);
        progress = other.progress;
        factor = other.factor;
        final = other.final;
        other.progress = nullptr;
        return *this;
    }

    bool isCancelled() const
    {
        return progress ? progress->isCancelled() : false;
    }

    void throwIfCancelled() const
    {
        if (progress)
            progress->throwIfCancelled();
    }

    void advance(float step)
    {
        if (progress)
            progress->advanceProgress(step * factor);
    }

    AsyncPartialProgress sub(float step)
    {
        return AsyncPartialProgress(progress, step);
    }
};

} // namespace hlp

#endif // ASYNCPROGRESS_H
