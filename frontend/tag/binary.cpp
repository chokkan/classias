/*
 *		Binary classification.
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

#ifdef  HAVE_CONFIG_H
#include <config.h>
#endif/*HAVE_CONFIG_H*/

#include <fstream>
#include <iostream>
#include <string>
#include <map>

#include <classias/classias.h>
#include <classias/classify/linear/binary.h>
#include <classias/evaluation.h>

#include "option.h"
#include "tokenize.h"
#include "defaultmap.h"
#include <util.h>

typedef defaultmap<std::string, double> model_type;
typedef classias::classify::linear_binary_logistic<model_type> classifier_type;

static void
parse_line(
    classifier_type& inst,
    bool& rl,
    const option& opt,
    const std::string& line,
    int lines = 0
    )
{
    double value;
    std::string name;

    // Split the line with tab characters.
    tokenizer values(line, opt.token_separator);
    tokenizer::iterator itv = values.begin();
    if (itv == values.end()) {
        throw invalid_data("no field found in the line", line, lines);
    }

    // The first field always presents a label, which can be empty.
    get_name_value(*itv, name, value, opt.value_separator);
    if (name == "+1" || name == "1") {
        rl = true;
    } else if (name == "-1") {
        rl = false;
    } else if (opt.test) {
        throw invalid_data("a class label must be either '+1', '1', or '-1'", line, lines);
    }

    // Apply the bias feature if any.
    inst.set("__BIAS__", 1.0);

    // Set featuress for the instance.
    for (++itv;itv != values.end();++itv) {
        if (!itv->empty()) {
            get_name_value(*itv, name, value, opt.value_separator);
            inst.set(name, value);
        }
    }
}

static void
read_model(
    model_type& model,
    std::istream& is,
    option& opt
    )
{
    for (;;) {
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }

        if (line.compare(0, 6, "@bias\t") == 0) {
            opt.bias = std::atof(line.c_str() + 6);
            continue;
        }

        int pos = line.find('\t');
        if (pos == line.npos) {
            throw invalid_model("feature weight is missing", line);
        }

        double w = std::atof(line.c_str());
        if (++pos == line.size()) {
            throw invalid_model("feature name is missing", line);
        }

        model[line.substr(pos)] = w;
    }
}

int binary_tag(option& opt, std::ifstream& ifs)
{
    int lines = 0;
    std::istream& is = opt.is;
    std::ostream& os = opt.os;
    classias::accuracy acc;
    classias::precall pr(2);

    // Load a model.
    model_type model;
    read_model(model, ifs, opt);

    for (;;) {
        // Read a line.
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }
        ++lines;

        // An empty line or comment line.
        if (line.empty() || line.compare(0, 1, "#") == 0) {
            // Output the comment line if necessary.
            if (opt.output & option::OUTPUT_COMMENT) {
                os << line << std::endl;
            }
            continue;
        }

        // Parse the line and classify the instance.
        bool rlabel;
        classifier_type inst(model);
        parse_line(inst, rlabel, opt, line, lines);

        // Output the label.
        if (opt.output & option::OUTPUT_MLABEL) {
            os << (static_cast<bool>(inst) ? "+1" : "-1");

            // Output the probability or score.
            if (opt.output & option::OUTPUT_PROBABILITY) {
                os << opt.value_separator << inst.prob();
            } else if (opt.output & option::OUTPUT_SCORE) {
                os << opt.value_separator << inst.score();
            }

            os << std::endl;
        }

        // Accumulate the performance.
        if (opt.test) {
            int rl = static_cast<int>(rlabel);
            int ml = static_cast<int>(static_cast<bool>(inst));
            acc.set(ml == rl);
            pr.set(ml, rl);
        }
    }

    // Output the performance if necessary.
    if (opt.test) {
        int positive_labels[] = {1};
        acc.output(os);
        pr.output_micro(os, positive_labels, positive_labels+1);
    }

    return 0;
}
