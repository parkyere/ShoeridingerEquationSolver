#pragma once

// Wall-clock callback scheduler for the SDL main loop. after() fires once;
// every() repeats until cancel(). poll(now_ms) runs due callbacks on the loop
// thread; callbacks may schedule further work (the selftest arcs chain this
// way).

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace ses_shell {

class Scheduler {
public:
    void after(std::uint64_t delay_ms, std::function<void()> fn) {
        entries_.push_back({next_id_++, now_ + delay_ms, 0, std::move(fn)});
    }

    int every(std::uint64_t period_ms, std::function<void()> fn) {
        const int id = next_id_++;
        entries_.push_back({id, now_ + period_ms, period_ms, std::move(fn)});
        return id;
    }

    void cancel(int id) {
        for (Entry& e : entries_) {
            if (e.id == id) {
                e.fn = nullptr;  // reaped on the next poll
            }
        }
    }

    void poll(std::uint64_t now_ms) {
        now_ = now_ms;
        // Index loop: callbacks may push_back new entries (they run NEXT poll
        // at the earliest, so newly due work never runs inside this pass).
        const std::size_t n = entries_.size();
        for (std::size_t i = 0; i < n; ++i) {
            Entry& e = entries_[i];
            if (!e.fn || e.due > now_ms) {
                continue;
            }
            if (e.period_ms > 0) {
                e.due = now_ms + e.period_ms;
                e.fn();  // repeating: fires and re-arms
            } else {
                auto fn = std::move(e.fn);
                e.fn = nullptr;
                fn();  // one-shot: consumed
            }
        }
        std::erase_if(entries_, [](const Entry& e) { return !e.fn; });
    }

private:
    struct Entry {
        int id;
        std::uint64_t due;
        std::uint64_t period_ms;  // 0 = one-shot
        std::function<void()> fn;
    };
    std::vector<Entry> entries_;
    std::uint64_t now_ = 0;
    int next_id_ = 1;
};

}  // namespace ses_shell
