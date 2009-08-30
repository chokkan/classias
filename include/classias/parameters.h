/*
 *		Utilities for parameter exchange.
 *
 * Copyright (c) 2008,2009 Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id$ */

#ifndef __CLASSIAS_PARAMS_H__
#define __CLASSIAS_PARAMS_H__

#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace classias
{

/**
 * Exception class for \ref parameter_exchange.
 */
class unknown_parameter : public std::invalid_argument
{
public:
    /**
     * Constructs an exception object.
     *  @param  message     The error message.
     */
    explicit unknown_parameter(const std::string& message)
        : std::invalid_argument(message)
    {
    }
};



/**
 * A class for exchanging parameters.
 */
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
        /// The time stamp when the parameter is set.
        int         stamp;
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
    /// A time stamp.
    int stamp;

    /**
     * Constructs the object.
     */
    parameter_exchange() : stamp(0)
    {
    }

    /**
     * Destructs the object.
     */
    virtual ~parameter_exchange()
    {
    }

    /**
     * Defines an integer parameter.
     *  @param  name            The name of the parameter.
     *  @param  var             The pointer to the parameter variable.
     *  @param  defval          The default value of the parameter.
     *  @param  message         The help message for the parameter.
     */
    void init(
        const std::string& name,
        int* var,
        const int defval = 0,
        const std::string& message = ""
        )
    {
        *var = defval;

        if (pmap.find(name) == pmap.end()) {
            value_type v;
            v.type = VT_INT;
            v.pointer = var;
            v.stamp = 0;
            v.message = message;
            pmap.insert(parameter_map::value_type(name, v));
            plist.push_back(name);
        }
    }

    /**
     * Defines a double parameter.
     *  @param  name            The name of the parameter.
     *  @param  var             The pointer to the parameter variable.
     *  @param  defval          The default value of the parameter.
     *  @param  message         The help message for the parameter.
     */
    void init(
        const std::string& name,
        double* var,
        const double defval = 0,
        const std::string& message = ""
        )
    {
        *var = defval;

        if (pmap.find(name) == pmap.end()) {
            value_type v;
            v.type = VT_DOUBLE;
            v.pointer = var;
            v.stamp = 0;
            v.message = message;
            pmap.insert(parameter_map::value_type(name, v));
            plist.push_back(name);
        }
    }

    /**
     * Defines a string parameter.
     *  @param  name            The name of the parameter.
     *  @param  var             The pointer to the parameter variable.
     *  @param  defval          The default value of the parameter.
     *  @param  message         The help message for the parameter.
     */
    void init(
        const std::string& name,
        std::string* var,
        const std::string& defval = "",
        const std::string& message = ""
        )
    {
        *var = defval;

        if (pmap.find(name) == pmap.end()) {
            value_type v;
            v.type = VT_STRING;
            v.pointer = var;
            v.stamp = 0;
            v.message = message;
            pmap.insert(parameter_map::value_type(name, v));
            plist.push_back(name);
        }
    }

    /**
     * Sets a value of a parameter.
     *  @param  name            The name of the parameter.
     *  @param  value           The integer value.
     *  @param  unk             The indicator specifying whether this function
     *                          throws an exception \ref unknown_parameter
     *                          when the parameter is undefined.
     */
    void set(const std::string& name, const int value, bool unk=true)
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
            it->second.stamp = ++stamp;
        } else if (unk) {
            throw unknown_parameter(name);
        }
    }

    /**
     * Sets a value of a parameter.
     *  @param  name            The name of the parameter.
     *  @param  value           The double value.
     *  @param  unk             The indicator specifying whether this function
     *                          throws an exception \ref unknown_parameter
     *                          when the parameter is undefined.
     */
    void set(const std::string& name, const double value, bool unk=true)
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
            it->second.stamp = ++stamp;
        } else if (unk) {
            throw unknown_parameter(name);
        }
    }

    /**
     * Sets a value of a parameter.
     *  @param  name            The name of the parameter.
     *  @param  value           The string value.
     *  @param  unk             The indicator specifying whether this function
     *                          throws an exception \ref unknown_parameter
     *                          when the parameter is undefined.
     */
    void set(const std::string& name, const std::string& value, bool unk=true)
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
            it->second.stamp = ++stamp;
        } else if (unk) {
            throw unknown_parameter(name);
        }
    }

    /**
     * Gets the time stamp when the parameter value is set.
     *  @param  name            The name of the parameter.
     *  @return int             The time stamp when the value of the parameter
     *                          is set.
     */
    int get_stamp(const std::string& name)
    {
        parameter_map::iterator it = pmap.find(name);
        return (it != pmap.end() ? it->second.stamp : -1);
    }

    /**
     * Shows the parameter values.
     *  @param  os              The output stream.
     *  @return std::ostream&   The output stream.
     */
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

    /**
     * Shows the help message of parameters.
     *  @param  os              The output stream.
     *  @return std::ostream&   The output stream.
     */
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
