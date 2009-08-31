/*
 *		Frontend for tagging.
 *
 * Copyright (c) 2008, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Northwestern University, University of Tokyo,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written
 *       permission.
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

#ifdef  HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <fstream>
#include <iostream>
#include <typeinfo>
#include <classias/version.h>
#include <optparse.h>

#include "option.h"

int binary_tag(option& opt, std::ifstream& ifs);
int multi_dense_tag(option& opt, std::ifstream& ifs);
int multi_sparse_tag(option& opt, std::ifstream& ifs);
int candidate_tag(option& opt, std::ifstream& ifs);

class optionparser : public option, public optparse
{
public:
    optionparser(
        std::istream& _is = std::cin,
        std::ostream& _os = std::cout,
        std::ostream& _es = std::cerr
        ) : option(_is, _os, _es)
    {
    }

    BEGIN_OPTION_MAP_INLINE()
        ON_OPTION_WITH_ARG(SHORTOPT('m') || LONGOPT("model"))
            model = arg;

        ON_OPTION(SHORTOPT('t') || LONGOPT("test"))
            test = true;

        ON_OPTION_WITH_ARG(SHORTOPT('s') || LONGOPT("token-separator"))
            if (strcmp(arg, " ") == 0 || strcasecmp(arg, "s") == 0 || strcasecmp(arg, "spc") == 0 || strcasecmp(arg, "space") == 0) {
                token_separator = ' ';
            } else if (strcmp(arg, ",") == 0 || strcasecmp(arg, "c") == 0 || strcasecmp(arg, "comma") == 0) {
                token_separator = ',';
            } else if (strcmp(arg, "\t") == 0 || strcasecmp(arg, "t") == 0 || strcasecmp(arg, "tab") == 0) {
                token_separator = '\t';
            } else {
                std::stringstream ss;
                ss << "unknown token separator specified: " << arg;
                throw invalid_value(ss.str());
            }

        ON_OPTION_WITH_ARG(SHORTOPT('c') || LONGOPT("value-separator"))
            if (strcmp(arg, ":") == 0 || strcasecmp(arg, "c") == 0 || strcasecmp(arg, "colon") == 0) {
                value_separator = ':';
            } else if (strcmp(arg, "=") == 0 || strcasecmp(arg, "e") == 0 || strcasecmp(arg, "eq") == 0 || strcasecmp(arg, "equal") == 0) {
                value_separator = '=';
            } else if (strcmp(arg, "|") == 0 || strcasecmp(arg, "b") == 0 || strcasecmp(arg, "bar") == 0) {
                value_separator = '|';
            } else {
                std::stringstream ss;
                ss << "unknown value separator specified: " << arg;
                throw invalid_value(ss.str());
            }

        ON_OPTION(SHORTOPT('k') || LONGOPT("comment"))
            output |= OUTPUT_COMMENT;

        ON_OPTION(SHORTOPT('q') || LONGOPT("quiet"))
            output = OUTPUT_NONE;

        ON_OPTION(SHORTOPT('w') || LONGOPT("score"))
            output |= OUTPUT_SCORE;

        ON_OPTION(SHORTOPT('p') || LONGOPT("probability"))
            output |= OUTPUT_PROBABILITY;

        ON_OPTION(SHORTOPT('v') || LONGOPT("version"))
            mode = MODE_VERSION;

        ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
            mode = MODE_HELP;

    END_OPTION_MAP()
};

static void usage(std::ostream& os, const char *argv0)
{
    os << "USAGE: " << argv0 << " [OPTIONS]" << std::endl;
    os << "This utility tags labels for a data set read from STDIN." << std::endl;
    os << std::endl;
    os << "OPTIONS:" << std::endl;
    os << "  -m, --model=FILE      load the model from FILE" << std::endl;
    os << "  -t, --test            evaluate the tagging performance on the labeled data" << std::endl;
    os << "  -s, --token-separator=SEP assume SEP character as a token separator:" << std::endl;
    os << "      ' ',  s, spc, space       a SPACE (' ') character (DEFAULT)" << std::endl;
    os << "      '\\t', t, tab              a TAB ('\\t') character" << std::endl;
    os << "      ',',  c, comma            a COMMA (',') character" << std::endl;
    os << "  -c, --value-separator=SEP assume SEP character as a value separator:" << std::endl;
    os << "      ':',  c, colon            a COLON (':') character (DEFAULT)" << std::endl;
    os << "      '=',  e, equal            a EQUAL ('=') character" << std::endl;
    os << "      '|',  b, bar              a BAR ('|') character" << std::endl;
    os << "  -w, --score           output scores for the labels" << std::endl;
    os << "  -p, --probability     output probabilities for the labels" << std::endl;
    os << "  -k, --comment         output commentlines for the tagging output" << std::endl;
    os << "  -q, --quiet           suppress tagging results from the output" << std::endl;
    os << "  -v, --version         show the version and copyright information" << std::endl;
    os << "  -h, --help            show this help message and exit" << std::endl;
    os << std::endl;
}

static int check_model(std::istream& is)
{
    // Read the first line of the model.
    std::string line;
    std::getline(is, line);

    if (line == "@classias\tlinear\tbinary") {
        return option::TYPE_BINARY;
    } else if (line == "@classias\tlinear\tmulti\tdense") {
        return option::TYPE_MULTI_DENSE;
    } else if (line == "@classias\tlinear\tmulti\tsparse") {
        return option::TYPE_MULTI_SPARSE;
    } else if (line == "@classias\tlinear\tcandidate") {
        return option::TYPE_CANDIDATE;
    } else {
        return option::TYPE_NONE;
    }
}

int main(int argc, char *argv[])
{
    int ret = 0;
    int arg_used = 0;
    optionparser opt;
    std::istream& is = opt.is;
    std::ostream& os = opt.os;
    std::ostream& es = opt.es;

    // Parse the command-line options.
    try { 
        arg_used = opt.parse(argv, argc);
    } catch (const optparse::unrecognized_option& e) {
        es << "ERROR: unrecognized option: " << e.what() << std::endl;
        return 1;
    } catch (const optparse::invalid_value& e) {
        es << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    // Show the help message and exit.
    if (opt.mode == option::MODE_HELP) {
        usage(os, argv[0]);
        return ret;
    } else if (opt.mode == option::MODE_VERSION) {
        // Show the copyright information.
        os << CLASSIAS_NAME " ";
        os << CLASSIAS_MAJOR_VERSION << "." << CLASSIAS_MINOR_VERSION << " ";
        os << "tagger ";
        os << CLASSIAS_COPYRIGHT << std::endl;
        os << std::endl;
        return ret;
    }

    // Set the source files.
    for (int i = arg_used;i < argc;++i) {
        opt.files.push_back(argv[i]);
    }

    // Open the model file.
    std::ifstream ifs(opt.model.c_str());
    if (ifs.fail()) {
        es << "ERROR: failed to open the model file: " << opt.model << std::endl;
        return 1;
    }

    // Check the model type.
    int type = check_model(ifs);

    try {
        // Branches for the model type.
        switch (type) {
        case option::TYPE_BINARY:
            ret = binary_tag(opt, ifs);
            break;
        case option::TYPE_MULTI_SPARSE:
        case option::TYPE_MULTI_DENSE:
            //ret = multi_train(opt);
            break;
        case option::TYPE_CANDIDATE:
            //ret = candidate_train(opt);
            break;
        default:
            es << "ERROR: unknown model type" << std::endl;
            ret = 1;
            break;
        }
    } catch (const std::exception& e) {
        es << "ERROR: " << typeid(e).name() << ": " << e.what() << std::endl;
        return 1;
    }

    return ret;
}
