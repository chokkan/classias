/*
 *		Classias frontend.
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

/* $Id:$ */

#ifdef  HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <iostream>
#include "option.h"
#include "optparse.h"
#include "util.h"
#include "tokenize.h"

#define	APPLICATION_S	"Classias"
#define	VERSION_S		"0.2"
#define	COPYRIGHT_S		"Copyright (c) 2008 Naoaki Okazaki"

int binary_train(option& opt);
bool binary_usage(option& opt);
int multi_train(option& opt);
bool multi_usage(option& opt);

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
        ON_OPTION(SHORTOPT('l') || LONGOPT("learn"))
            mode = MODE_TRAIN;

        ON_OPTION(SHORTOPT('t') || LONGOPT("tag"))
            mode = MODE_TAG;

        ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
            mode = MODE_HELP;

        ON_OPTION(SHORTOPT('H') || LONGOPT("help-parameters"))
            mode = MODE_HELP_ALGORITHM;

        ON_OPTION_WITH_ARG(SHORTOPT('f') || LONGOPT("task"))
            if (strcmp(arg, "binary") == 0 || strcmp(arg, "b") == 0) {
                type = TYPE_BINARY;
            } else if (strcmp(arg, "multi") == 0 || strcmp(arg, "m") == 0) {
                type = TYPE_MULTI;
            } else {
                std::stringstream ss;
                ss << "unknown task type specified: " << arg;
                throw invalid_value(ss.str());
            }

        ON_OPTION_WITH_ARG(SHORTOPT('m') || LONGOPT("model"))
            model = arg;

        ON_OPTION_WITH_ARG(LONGOPT("negative"))
            negatives.clear();
            std::string labels = arg;
            tokenizer field(labels, ' ');
            while (field.next()) {
                if (!field->empty()) {
                    negatives.insert(*field);
                }
            }

        ON_OPTION_WITH_ARG(SHORTOPT('a') || LONGOPT("algorithm"))
            if (strcmp(arg, "maxent") == 0) {
            } else if (strcmp(arg, "logress") == 0) {
            } else {
                std::stringstream ss;
                ss << "unknown training algorithm specified: " << arg;
                throw invalid_value(ss.str());
            }
            algorithm = arg;

        ON_OPTION(SHORTOPT('b') || LONGOPT("generate-bias"))
            generate_bias = true;

        ON_OPTION_WITH_ARG(SHORTOPT('p') || LONGOPT("set"))
            params.push_back(arg);

        ON_OPTION_WITH_ARG(SHORTOPT('g') || LONGOPT("split"))
            split = atoi(arg);

        ON_OPTION_WITH_ARG(SHORTOPT('e') || LONGOPT("holdout"))
            holdout = atoi(arg);

        ON_OPTION(SHORTOPT('x') || LONGOPT("cross-validate"))
            cross_validation = true;

    END_OPTION_MAP()
};

static void usage(std::ostream& os, const char *argv0)
{
    os << "USAGE: " << argv0 << " [OPTIONS] [DATA1] [DATA2] ..." << std::endl;
    os << "  DATA    file(s) corresponding to a data set for the processing;" << std::endl;
    os << "          if multiple N files are specified, this tool assumes a data set to" << std::endl;
    os << "          be split into N groups and assigns a group number (1...N) to the" << std::endl;
    os << "          instances in each file; if no file is specified, the tool reads a" << std::endl;
    os << "          data set from STDIN" << std::endl;
    os << std::endl;
    os << "COMMANDS:" << std::endl;
    os << "  -l, --learn           train a model from the training set" << std::endl;
    os << "  -t, --tag             tag the data with the model (specified by -m option)" << std::endl;
    os << "  -h, --help            show the help message and exit" << std::endl;
    os << "  -H, --help-parameters show the help message of parameters for the algorithm" << std::endl;
    os << "                        specified by the -a option" << std::endl;
    os << std::endl;
    os << "COMMON OPTIONS:" << std::endl;
    os << "  -f, --task=TYPE       specify a task type (DEFAULT='multi'):" << std::endl;
    os << "      b, binary             an instance consists of a feature vector and a" << std::endl;
    os << "                            boolean class, +1 or -1" << std::endl;
    os << "      m, multi              an instance consists of multiple candidates each of" << std::endl;
    os << "                            which has an feature vector, boolean class, and" << std::endl;
    os << "                            label (optional)" << std::endl;
    os << "  -m, --model=FILE      store/load a model to/from FILE (DEFAULT=''); if the" << std::endl;
    os << "                        value is empty, this tool does not store/load a model" << std::endl;
    os << "      --negative=LABELS specify negative LABELS (separated by space characters)" << std::endl;
    os << "                        (DEFAULT='-1 O'); this tool assumes LABELS as negative" << std::endl;
    os << "                        instances when computing precision/recall/f1-score" << std::endl;
    os << std::endl;
    os << "TRAINING OPTIONS:" << std::endl;
    os << "  -a, --algorithm=NAME  specify a training algorithm (DEFAULT='maxent')" << std::endl;
    os << "      maxent                maximum entropy model (for multi)" << std::endl;
    os << "      logress               logistic regression model (for binary)" << std::endl;
    os << "  -b, --generate-bias   generate bias features automatically" << std::endl;
    os << "  -p, --set=NAME=VALUE  set the algorithm-specific parameter NAME to VALUE;" << std::endl;
    os << "                        enter '-h' or '--help' followed by the algorithm name" << std::endl;
    os << "                        to see the documentation for the parameters" << std::endl;
    os << "  -g, --split=N         split the instances into N groups; this option is" << std::endl;
    os << "                        useful for holdout evaluation and cross validation" << std::endl;
    os << "  -e, --holdout=M       use the M-th data for holdout evaluation and the rest" << std::endl;
    os << "                        for training" << std::endl;
    os << "  -x, --cross-validate  repeat holdout evaluations for #i in {1, ..., N}" << std::endl;
    os << "                        (N-fold cross validation)" << std::endl;
    os << std::endl;
}


int main(int argc, char *argv[])
{
    int ret = 0;
    int arg_used = 0;
    optionparser opt;
    std::istream& is = opt.is;
    std::ostream& os = opt.os;
    std::ostream& es = opt.es;

    // Show the copyright information.
    es << APPLICATION_S " " VERSION_S "  " COPYRIGHT_S << std::endl;
    es << std::endl;

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
    } else if (opt.mode == option::MODE_HELP_ALGORITHM) {
        multi_usage(opt) || binary_usage(opt);
        return ret;
    }

    // Set the source files.
    for (int i = arg_used;i < argc;++i) {
        opt.files.push_back(argv[i]);
    }

    // Branch for tasks.
    try {
        if (opt.mode == option::MODE_TRAIN) {
            switch (opt.type) {
            case option::TYPE_BINARY:
                ret = binary_train(opt);
                break;
            case option::TYPE_MULTI:
                ret = multi_train(opt);
                break;
            }
        } else if (opt.mode == option::MODE_TAG) {

        }
    } catch (const std::exception& e) {
        es << "ERROR: " << typeid(e).name() << ": " << e.what() << std::endl;
        return 1;
    }

    return ret;
}
