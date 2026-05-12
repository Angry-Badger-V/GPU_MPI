#pragma once
#include <string>
#include <unordered_map>
#include <cstdlib>

struct Args {
    std::unordered_map<std::string,std::string> kv;

    static Args parse(int argc, char** argv) {
        Args a;
        for (int i = 1; i < argc; ++i) {
            std::string s(argv[i]);
            if (s.rfind("--", 0) == 0) {
                std::string key = s.substr(2);
                std::string val = "1";
                if (i + 1 < argc) {
                    std::string nxt(argv[i+1]);
                    if (nxt.rfind("--", 0) != 0) { val = nxt; ++i; }
                }
                a.kv[key] = val;
            }
        }
        return a;
    }

    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k); return it == kv.end() ? def : it->second;
    }
    long long get_ll(const std::string& k, long long def) const {
        auto it = kv.find(k); return it == kv.end() ? def : std::atoll(it->second.c_str());
    }
    int get_int(const std::string& k, int def) const {
        auto it = kv.find(k); return it == kv.end() ? def : std::atoi(it->second.c_str());
    }
    double get_double(const std::string& k, double def) const {
        auto it = kv.find(k); return it == kv.end() ? def : std::atof(it->second.c_str());
    }
};
