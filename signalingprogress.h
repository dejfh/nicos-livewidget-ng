//#ifndef SIGNALINGPROGRESS_H
//#define SIGNALINGPROGRESS_H

//#include <QObject>
//#include <atomic>

//#include "asyncprogress.h"

//class SignalingProgress : public QObject, AsyncProgressMaster
//{
//    Q_OBJECT

//    std::atomic<float> m_progress;
//    std::atomic<float> m_aborted;

//    int m_factor;

//public:
//    SignalingProgress(QObject *parent = 0, int factor = 10000);

//signals:
//    void progressChanged(int);

//public:
//    void abort();
//    double progress() const;

//    bool canceled() const;
//    bool setProgress(float progress);

//    AsyncFloatProgress subProgress(float amount = 1.0);
//};

//#endif // SIGNALINGPROGRESS_H
