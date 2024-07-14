#ifndef RCU_NOWAIT_H_
#define RCU_NOWAIT_H_
/*
 * RCU no wait
 * https://github.com/lano1106/rcu_nowait
 *
 * Small abstraction to offer RCU like functionality minus the grace period.
 * Unless I misunderstood how RCU works, when a publishing thread is ready to
 * update the protected data, a pointer is replaced atomically.
 *
 * In parallel to that, old data incarnations are tolerated during a short
 * period of time (grace period) to allow consumers thread to finish referring
 * to them before the publishing thread dispose them. This imply that the
 * publishing has to wait in order to offer a nowait structure to reading
 * threads. This is the type of compromise that cannot be done in my
 * application but the RCU concept is very appealing.
 *
 * Olivier Langlois - June 25, 2024
 */

#include <algorithm> // for std::copy()
#include <array>
#include <atomic>
#include <memory>

#ifndef likely
#define likely(x)	__builtin_expect (!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x)	__builtin_expect (!!(x), 0)
#endif

namespace Base {

/*
 * class RcuDataItem
 *
 * It is a base class implementation of the required concept for RcuDataNoWait
 * T parameter. It is not mandatory to use this made-for-you-and-ready-to-use
 * implementation in your T but why not?
 */
class RcuDataItem
{
public:
    RcuDataItem() = default;
    // ref counter are not copied.
    RcuDataItem(const RcuDataItem &rhs) {}
    RcuDataItem &operator=(const RcuDataItem &rhs)
    { return *this; }

    void release() const
    { m_refCounter.fetch_sub(1, std::memory_order::release); }
    void acquire()
    { m_refCounter.fetch_add(1, std::memory_order::release); }
    long queryRefCount() const
    { return m_refCounter.load(std::memory_order::acquire); }

private:
    /*
     * NOTE:
     * If the derived class is at least 64 bytes of size,
     * this ensure that each atomic variable has its own cacheline
     * and that false sharing will not happen.
     *
     * If not, this is something to look into for optimal performance.
     */
    mutable std::atomic_long m_refCounter{};
};

/*
 * class RcuDataNoWait
 *
 * BUFSZ requirements:
 * - It has to be at least 'maximum number of concurrent reading threads' + 2 (or
 *   more but the smallest needed size is best). To simplify pointer arithmetic,
 * - BUFSZ must also be a power of 2.
 */
template <class T, size_t BUFSZ>
class RcuDataNoWait
{
    static_assert(BUFSZ != 0 && (BUFSZ & (BUFSZ - 1)) == 0,
                  "RcuDataNoWait<BUFSZ> must be a power of 2");

public:
    template<class InputIterator>
    RcuDataNoWait(InputIterator first, InputIterator last)
    { std::copy(first, last, std::begin(m_data)); }

    /*
     * increase refCount of the element at the head and wrap it
     * into a std::unique_ptr to make sure that the refCount gets decremented
     * when done with it.
     */
    auto read() const
    {
        auto relFunc{[](const T *ptr){
            ptr->release();
        }};
        std::unique_ptr<const T,decltype(relFunc)> ptr{nullptr, relFunc};
        auto curHead{m_head.load(std::memory_order::acquire)};
        decltype(curHead) oldHead;

        do {
            T *rawP{m_data[curHead&(BUFSZ-1)]};

            rawP->acquire();
            ptr.reset(rawP);
            oldHead = curHead;
        } while (unlikely((curHead = m_head.load(std::memory_order::acquire)) != oldHead));

        return ptr;
    }

    /*
     * find the first element with an empty count.
     * swap it with the tail if not the tail and return it.
     */
    T *initUpdate()
    {
        auto nextPos{m_head.load(std::memory_order::relaxed)};
        auto *first{&m_data[(++nextPos)&(BUFSZ-1)]};

        if (unlikely((*first)->queryRefCount() != 0)) {
            auto *it{&m_data[(++nextPos)&(BUFSZ-1)]};

            while ((*it)->queryRefCount() != 0) it = &m_data[(++nextPos)&(BUFSZ-1)];
            std::swap(*first, *it);
        }
        return *first;
    }
    /*
     * returns the head for initializing the upcoming update by copying its
     * content into the updated item.
     *
     * there is no need to synchronize with m_head since it is the publisher that
     * updates it so no safety issue here.
     */
    const T *publisherRead() const
    {
        auto curHead{m_head.load(std::memory_order::relaxed)};

        return m_data[curHead&(BUFSZ-1)];
    }
    /*
     * this function is called to install the freshly updated item into the
     * head
     */
    void commitUpdate()
    { m_head.fetch_add(1, std::memory_order::release); }

private:
    std::array<T*,BUFSZ> m_data;
    std::atomic_long     m_head{};
};

}

#endif
