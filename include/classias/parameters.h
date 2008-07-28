#ifndef __CLASSIAS_PARAMS_H__
#define __CLASSIAS_PARAMS_H__

#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace classias
{

class unknown_parameter : public std::invalid_argument
{
public:
    explicit unknown_parameter(const std::string& message)
        : std::invalid_argument(message)
    {
    }
};


class parameter_exchange
{
public:
    /// Parameter types.
    enum {
        /// Parameter type \c int .
        VT_INT,
        /// Parameter type \c double .
        VT_DOUBLE,
        /// Parameter type \c std::string .
        VT_STRING,
    };

    /// Parameter value.
    struct value_type
    {
        /// Parameter type.
        int         type;
        /// The pointer to the parameter.
        void*       pointer;
        /// The help message.
        std::string message;
    };

    /// A type providing a mapping from parameter names to their values.
    typedef std::map<std::string, value_type> parameter_map;
    /// A type providing a list of parameter names.
    typedef std::vector<std::string> parameter_list;

    /// A parameter map.
    parameter_map   pmap;
    /// A parameter list.
    parameter_list  plist;

    /**
     * Constructs the object.
     */
    parameter_exchange()
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~parameter_exchange()
    {
    }

    void init(const std::string& name, int* var, const int defval = 0, const std::string& message = "")
    {
        *var = defval;

        if (pmap.find(name) == pmap.end()) {
            value_type v;
            v.type = VT_INT;
            v.pointer = var;
            v.message = message;
            pmap.insert(parameter_map::value_type(name, v));
            plist.push_back(name);
        }
    }

    void init(const std::string& name, double* var, const double defval = 0, const std::string& message = "")
    {
        *var = defval;

        if (pmap.find(name) == pmap.end()) {
            value_type v;
            v.type = VT_DOUBLE;
            v.pointer = var;
            v.message = message;
            pmap.insert(parameter_map::value_type(name, v));
            plist.push_back(name);
        }
    }

    void init(const std::string& name, std::string* var, const std::string& defval = "", const std::string& message = "")
    {
        *var = defval;

        if (pmap.find(name) == pmap.end()) {
            value_type v;
            v.type = VT_STRING;
            v.pointer = var;
            v.message = message;
            pmap.insert(parameter_map::value_type(name, v));
            plist.push_back(name);
        }
    }

    void set(const std::string& name, const int value)
    {
        parameter_map::iterator it = pmap.find(name);
        if (it != pmap.end()) {
            if (it->second.type == VT_INT) {
                *reinterpret_cast<int*>(it->second.pointer) = value;
            } else if (it->second.type == VT_DOUBLE) {
                *reinterpret_cast<double*>(it->second.pointer) = (double)value;
            } else if (it->second.type == VT_STRING) {
                std::stringstream ss;
                ss << value;
                *reinterpret_cast<std::string*>(it->second.pointer) = ss.str();
            }
        } else {
            throw unknown_parameter(name);
        }
    }

    void set(const std::string& name, const double value)
    {
        parameter_map::iterator it = pmap.find(name);
        if (it != pmap.end()) {
            if (it->second.type == VT_INT) {
                *reinterpret_cast<int*>(it->second.pointer) = (int)value;
            } else if (it->second.type == VT_DOUBLE) {
                *reinterpret_cast<double*>(it->second.pointer) = value;
            } else if (it->second.type == VT_STRING) {
                std::stringstream ss;
                ss << value;
                *reinterpret_cast<std::string*>(it->second.pointer) = ss.str();
            }
        } else {
            throw unknown_parameter(name);
        }
    }

    void set(const std::string& name, const std::string& value)
    {
        parameter_map::iterator it = pmap.find(name);
        if (it != pmap.end()) {
            if (it->second.type == VT_INT) {
                *reinterpret_cast<int*>(it->second.pointer) = std::atoi(value.c_str());
            } else if (it->second.type == VT_DOUBLE) {
                *reinterpret_cast<double*>(it->second.pointer) = std::atof(value.c_str());
            } else if (it->second.type == VT_STRING) {
                *reinterpret_cast<std::string*>(it->second.pointer) = value;
            }
        } else {
            throw unknown_parameter(name);
        }
    }

    std::ostream& show(std::ostream& os)
    {
        parameter_list::const_iterator it;
        for (it = plist.begin();it != plist.end();++it) {
            parameter_map::const_iterator itp = pmap.find(*it);
            if (itp != pmap.end()) {
                const int type = itp->second.type;
                const void* pointer = itp->second.pointer;

                os << *it << ": ";
                if (type == VT_INT) {
                    os << *reinterpret_cast<const int*>(pointer);
                } else if (type == VT_DOUBLE) {
                    os << *reinterpret_cast<const double*>(pointer);
                } else if (type == VT_STRING) {
                    os << *reinterpret_cast<const std::string*>(pointer);
                }
                os << std::endl;
            }
        }

        return os;
    }

    std::ostream& help(std::ostream& os)
    {
        parameter_list::const_iterator it;
        for (it = plist.begin();it != plist.end();++it) {
            parameter_map::const_iterator itp = pmap.find(*it);
            if (itp != pmap.end()) {
                const int type = itp->second.type;
                const void* pointer = itp->second.pointer;

                os << itp->second.message << std::endl;
                os << "   ";
                if (type == VT_INT) {
                    os << "int    " << *it << " = " <<
                        *reinterpret_cast<const int*>(pointer) << std::endl;
                } else if (type == VT_DOUBLE) {
                    os << "double " << *it << " = " <<
                        *reinterpret_cast<const double*>(pointer) << std::endl;
                } else if (type == VT_STRING) {
                    os << "string " << *it << " = " <<
                        "'" << *reinterpret_cast<const std::string*>(pointer) << "'" << std::endl;
                }
                os << std::endl;
            }
        }

        return os;
    }
};

};

#endif/*__CLASSIAS_PARAMS_H__*/
