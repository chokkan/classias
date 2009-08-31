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

class option
{
public:
    typedef std::set<std::string> labelset_type;

public:
    enum {
        MODE_NORMAL = 0,        /// Normal mode.
        MODE_VERSION,           /// Version mode.
        MODE_HELP,              /// Usage mode.
    };

    enum {
        TYPE_NONE = 0,      /// Default type.
        TYPE_BINARY,        /// Binary classification.
        TYPE_MULTI_SPARSE,  /// Attribute-label classification.
        TYPE_MULTI_DENSE,   /// Attribute-label with dense features.
        TYPE_CANDIDATE,     /// Multi-candidate ranker.
    };

    enum {
        OUTPUT_NONE =           0x0000,
        OUTPUT_MLABEL =         0x0001,
        OUTPUT_COMMENT =        0x0004,
        OUTPUT_SCORE =          0x0010,
        OUTPUT_PROBABILITY =    0x0020,
    };

    std::istream&   is;
    std::ostream&   os;
    std::ostream&   es;

    int         mode;
    std::string model;
    bool        test;
    bool        false_analysis;
    int         output;

    char        token_separator;
    char        value_separator;

    labelset_type   negative_labels;

    option(
        std::istream& _is = std::cin,
        std::ostream& _os = std::cout,
        std::ostream& _es = std::cerr
        ) :
        is(_is), os(_os), es(_es),
        mode(MODE_NORMAL),
        test(false), false_analysis(false), output(OUTPUT_MLABEL),
        token_separator(' '), value_separator(':')
    {
    }
};

#endif/*__OPTION_H__*/
