/*
 *		Multi-class classifier.
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
#include <iterator>
#include <string>

#include <classias/classias.h>
#include <classias/quark.h>
#include <classias/classify/linear/multi.h>
#include <classias/evaluation.h>

#include "option.h"
#include "tokenize.h"
#include "defaultmap.h"
#include <util.h>

typedef defaultmap<std::string, double> model_type;
typedef std::vector<std::string> labels_type;
typedef std::vector<int> positive_labels_type;
typedef classias::classify::linear_multi_logistic<model_type> classifier_type;

class feature_generator
{
public:
    typedef std::string attribute_type;
    typedef std::string label_type;
    typedef std::string feature_type;

public:
    feature_generator()
    {
    }

    virtual ~feature_generator()
    {
    }

    inline bool forward(
        const std::string& a,
        const std::string& l,
        std::string& f
        ) const
    {
        f  = a;
        f += '\t';
        f += l;
        return true;
    }
};

static void
parse_line(
    classifier_type& inst,
    const feature_generator& fgen,
    std::string& rl,
    const classias::quark& labels,
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

    // Make sure that the first token (class) is not empty.
    if (itv->empty()) {
        throw invalid_data("an empty label found", line, lines);
    }

    // Parse the instance label.
    get_name_value(*itv, name, value, opt.value_separator);
    rl = name;

    // Initialize the classifier.
    inst.resize(labels.size());
    inst.clear();

    // Set attributes for the instance.
    for (++itv;itv != values.end();++itv) {
        if (!itv->empty()) {
            double value;
            std::string name;
            get_name_value(*itv, name, value, opt.value_separator);

            for (int i = 0;i < (int)labels.size();++i) {
                inst.set(i, fgen, name, labels.to_item(i), value);
            }
        }
    }

    // Apply the bias feature if any.
    for (int i = 0;i < (int)labels.size();++i) {
        inst.set(i, fgen, "__BIAS__", labels.to_item(i), value);
    }

    // Finalize the instance.
    inst.finalize();
}

static void
read_model(
    model_type& model,
    classias::quark& labels,
    std::istream& is,
    const option& opt
    )
{
    for (;;) {
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }

        // Candidate label.
        if (line.compare(0, 7, "@label\t") == 0) {
            labels(line.substr(7));
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

int multi_tag(option& opt, std::ifstream& ifs)
{
    int lines = 0;
    feature_generator fgen;
    std::istream& is = opt.is;
    std::ostream& os = opt.os;

    // Load a model.
    model_type model;
    classias::quark labels;
    read_model(model, labels, ifs, opt);

    // Create an instance of a classifier on the model.
    classifier_type inst(model);

    // Generate a set of positive labels (necessary only for evaluation).
    positive_labels_type positives;
    if (opt.test) {
        for (int i = 0;i < (int)labels.size();++i) {
            if (opt.negative_labels.find(labels.to_item(i)) == opt.negative_labels.end()) {
                positives.push_back(i);
            }
        }
    }

    // Objects for performance evaluation.
    classias::accuracy acc;
    classias::precall pr(labels.size());

    for (;;) {
        // Read a line.
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }
        ++lines;

        // A comment line.
        if (line.compare(0, 1, "#") == 0) {
            // Output the comment line if necessary.
            if (opt.output & option::OUTPUT_COMMENT) {
                os << line << std::endl;
            }
            continue;
        }

        // Parse the line and classify the instance.
        std::string rlabel;
        parse_line(inst, fgen, rlabel, labels, opt, line, lines);

        // Output the label.
        if (opt.output & option::OUTPUT_MLABEL) {
            os << labels.to_item(inst.argmax());

            // Output the probability or score.
            if (opt.output & option::OUTPUT_PROBABILITY) {
                os << opt.value_separator << inst.prob(inst.argmax());
            } else if (opt.output & option::OUTPUT_SCORE) {
                os << opt.value_separator << inst.score(inst.argmax());
            }

            os << std::endl;
        }

        // Accumulate the performance.
        if (opt.test) {
            try {
                int rl = inst.argmax();
                int ml = labels.to_value(rlabel);
                acc.set(ml == rl);
                pr.set(ml, rl);
            } catch (classias::quark_error&) {
            }
        }
    }

    // Output the performance if necessary.
    if (opt.test) {
        acc.output(os);
        pr.output_micro(os, positives.begin(), positives.end());
        pr.output_macro(os, positives.begin(), positives.end());
    }

    return 0;
}
