/*
 *		Miscellaneous utilities.
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

#ifndef __UTIL_H__
#define __UTIL_H__

#include <ctime>
#include <sstream>
#include <string>
#include <exception>

class invalid_data : public std::exception
{
protected:
    std::string message;

public:
    invalid_data(const char *const& msg, int lines)
    {
        std::stringstream ss;
        ss << "in lines " << lines << ", " << msg;
        message = ss.str();
    }

    invalid_data(const invalid_data& rho)
    {
        message = rho.message;
    }

    invalid_data& operator=(const invalid_data& rho)
    {
        message = rho.message;
    }

    virtual ~invalid_data() throw()
    {
    }

    virtual const char *what() const throw()
    {
        return message.c_str();
    }
};

class invalid_algorithm : public std::domain_error
{
public:
    explicit invalid_algorithm(const std::string& msg)
        : std::domain_error(msg)
    {
    }
};

class stopwatch
{
protected:
    clock_t begin;
    clock_t end;

public:
    stopwatch()
    {
        start();
    }

    void start()
    {
        begin = end = std::clock();
    }

    double stop()
    {
        end = std::clock();
        return get();
    }

    double get() const
    {
        return (end - begin) / (double)CLOCKS_PER_SEC;
    }
};

static void
get_name_value(
    const std::string& str, std::string& name, double& value, char separator)
{
    size_t col = str.rfind(separator);
    if (col == str.npos) {
        name = str;
        value = 1.;
    } else {
        value = std::atof(str.c_str() + col + 1);
        name = str.substr(0, col);
    }
}

template <class char_type, class traits_type>
inline std::basic_ostream<char_type, traits_type>&
timestamp(std::basic_ostream<char_type, traits_type>& os)
{
	time_t ts;
	time(&ts);

   	char str[80];
    std::strftime(
        str, sizeof(str), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&ts));

    os << str;
    return (os);
}

#endif/*__UTIL_H__*/
