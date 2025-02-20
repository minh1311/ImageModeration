// /**
// * Create by QuangNguyen on 15/12/2023
// */

// #ifndef CERBERUSAISDK_TRACE_H
// #define CERBERUSAISDK_TRACE_H

// #include <memory>
// #include <deque>

// //#include "cerberus/types/Keypoint.h"
// //
// //typedef struct TrajectoryPoint
// //{
// //    TrajectoryPoint(float x, float y)
// //    {
// //        m_kpt.x = x;
// //        m_kpt.y = y;
// //    }
// //
// //    TrajectoryPoint(const cerberus::Keypoint & kpt) : m_kpt(kpt) {}
// //
// //    cerberus::Keypoint m_kpt;
// //
// //} TrajectoryPoint_t;

// class Trace
// {
// public:
//    const cerberus::Keypoint& operator[](size_t i) const
//    {
//        return m_trace[i].m_kpt;
//    }

//    cerberus::Keypoint& operator[](size_t i)
//    {
//        return m_trace[i].m_kpt;
//    }

//    const TrajectoryPoint_t& at(size_t i) const
//    {
//        return m_trace[i];
//    }

//    size_t size() const
//    {
//        return m_trace.size();
//    }

//    void push_back(const cerberus::Keypoint& kpt)
//    {
//        m_trace.emplace_back(kpt);
//    }

//    void pop_front(size_t count)
//    {
//        if (count < size())
//        {
//            m_trace.erase(m_trace.begin(), m_trace.begin() + count);
//        }
//        else
//        {
//            m_trace.clear();
//        }
//    }

// private:
//    std::deque<TrajectoryPoint_t> m_trace;
// };

// #endif // CERBERUSAISDK_TRACE_H