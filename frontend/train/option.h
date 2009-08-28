/*
 *		Processing options.
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

#ifndef __OPTION_H__
#define __OPTION_H__

#include <vector>
#include <set>
#include <string>

#if defined _MSC_VER
#include <regex>
#define REGEX std::tr1::regex
#define REGEX_SEARCH std::tr1::regex_search

#elif defined __GNUC__
#include <boost/regex.hpp>
#define REGEX boost::regex
#define REGEX_SEARCH boost::regex_search

#endif

class option
{
public:
    typedef std::vector<std::string>    files_type;
    typedef std::vector<std::string>    params_type;
    typedef std::set<std::string>       labels_type;

    enum {
        MODE_NORMAL = 0,        /// Normal mode.
        MODE_HELP,              /// Usage mode.
        MODE_HELP_ALGORITHM,    /// Algorithm-specific usage.
    };

    enum {
        TYPE_NONE = 0,      /// Default type.
        TYPE_BINARY,        /// Binary classification.
        TYPE_MULTI_SPARSE,  /// Attribute-label classification.
        TYPE_MULTI_DENSE,   /// Attribute-label with dense features.
        TYPE_CANDIDATE,     /// Multi-candidate ranker.
    };

    std::istream&   is;
    std::ostream&   os;
    std::ostream&   es;

    files_type  files;

    int         mode;
    int         type;
    std::string algorithm;
    params_type params;
    std::string model;
    bool        shuffle;
    bool        generate_bias;
    int         split;
    int         holdout;
    REGEX       filter;
    std::string filter_string;
    bool        cross_validation;
    labels_type negative_labels;

    char        token_separator;
    char        value_separator;

    option(
        std::istream& _is = std::cin,
        std::ostream& _os = std::cout,
        std::ostream& _es = std::cerr
        ) :
        is(_is), os(_os), es(_es),
        mode(MODE_NORMAL), type(TYPE_MULTI_DENSE), model(""),
        algorithm("logress.lbfgs"),        
        shuffle(false), generate_bias(false),
        split(0), holdout(-1), cross_validation(false),
        token_separator(' '), value_separator(':')
    {
    }
};

#endif/*__OPTION_H__*/
