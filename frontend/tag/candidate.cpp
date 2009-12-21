/*
 *		Candidate classifier.
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
#include <vector>

#include <classias/classias.h>
#include <classias/classify/linear/multi.h>
#include <classias/evaluation.h>

#include "option.h"
#include "tokenize.h"
#include "defaultmap.h"
#include <util.h>

typedef defaultmap<std::string, double> model_type;
typedef std::vector<std::string> labels_type;
typedef std::vector<std::string> comments_type;
typedef classias::classify::linear_multi_logistic<model_type> classifier_type;

class feature_generator
{
public:
    typedef std::string attribute_type;
    typedef int label_type;
    typedef std::string feature_type;

public:
    feature_generator()
    {
    }

    virtual ~feature_generator()
    {
    }

    inline bool forward(
        const attribute_type& a,
        const label_type& l,
        feature_type& f
        ) const
    {
        f = a;
        return true;
    }
};

static void
parse_line(
    classifier_type& inst,
    const feature_generator& fgen,
    std::string& label,
    bool& truth,
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

    // Set the truth value for this candidate.
    if (itv->compare(0, 1, "+") == 0) {
        truth = true;
    } else if (itv->compare(0, 1, "-") == 0) {
        truth = false;
    } else {
        throw invalid_data("a class label must begins with '+' or '-'", line, lines);
    }

    label = itv->substr(1);

    // Create a new candidate.
    int i = inst.size();
    inst.resize(i+1);

    // Set featuress for the instance.
    for (++itv;itv != values.end();++itv) {
        if (!itv->empty()) {
            get_name_value(*itv, name, value, opt.value_separator);
            inst.set(i, fgen, name, 0, value);
        }
    }
}

static void
read_model(
    model_type& model,
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

static void output_model_candidates(
    std::ostream& os,
    classifier_type& inst,
    const int rl,
    const labels_type& labels,
    const comments_type& comments,
    option& opt
    )
{
    // Output all candidates.
    for (int i = 0;i < inst.size();++i) {
        // Output the reference class.
        if (opt.output & option::OUTPUT_RLABEL) {
            os << ((i == rl) ? '+' : '-');
        }

        // Output the tagging result and label.
        if (opt.output & option::OUTPUT_MLABEL) {
            os << ((i == inst.argmax()) ? '+' : '-');
        }

        os << labels[i];

        // Output the probability or score.
        if (opt.output & option::OUTPUT_PROBABILITY) {
            os << opt.value_separator << inst.prob(i);
        } else if (opt.output & option::OUTPUT_SCORE) {
            os << opt.value_separator << inst.score(i);
        }

        os << std::endl;
        os << comments[i];
    }

    os << "@eoi" << std::endl;
}

static void output_model_label(
    std::ostream& os,
    classifier_type& inst,
    const labels_type& labels,
    const comments_type& comments,
    option& opt
    )
{
    // Output the argmax label.
    os << labels[inst.argmax()];

    // Output the probability or score.
    if (opt.output & option::OUTPUT_PROBABILITY) {
        os << opt.value_separator << inst.prob(inst.argmax());
    } else if (opt.output & option::OUTPUT_SCORE) {
        os << opt.value_separator << inst.score(inst.argmax());
    }

    os << std::endl;
}

int candidate_tag(option& opt, std::ifstream& ifs)
{
    int rl = -1;
    int lines = 0;
    feature_generator fgen;
    std::istream& is = opt.is;
    std::ostream& os = opt.os;
    bool inner = false;
    std::string comment_outer, comment_inner;
    comments_type comments;

    // Load a model.
    model_type model;
    read_model(model, ifs, opt);

    // Create an instance of a classifier on the model.
    classifier_type inst(model);
    labels_type labels;

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

        // An empty line or comment line.
        if (line.empty() || line.compare(0, 1, "#") == 0) {
            if (opt.output & option::OUTPUT_COMMENT) {
                if (0 < inst.size()) {
                    // Store the comment line to the current instance.
                    comments[inst.size()-1] += line;
                    comments[inst.size()-1] += '\n';
                } else if (inner) {
                    comment_inner += line;
                    comment_inner += '\n';
                } else {
                    comment_outer += line;
                    comment_outer += '\n';
                }
            }
            continue;
        }

        if (line.compare(0, 4, "@boi") == 0) {
            // Begin of an instance.
            rl = -1;
            inst.clear();
            labels.clear();
            comments.clear();
            inner = true;

        } else if (line == "@eoi") {
            inst.finalize();

            // Determine whether we output this instance or not.
            if (opt.condition == option::CONDITION_ALL ||
                (opt.condition == option::CONDITION_FALSE && rl != inst.argmax())) {

                // Output BOI.
                os << comment_outer;
                os << "@boi" << std::endl;
                os << comment_inner;

                if (opt.output & option::OUTPUT_ALL) {
                    for (int i = 0;i < inst.size();++i) {
                        // Output the reference label.
                        if (opt.output & option::OUTPUT_RLABEL) {
                            os << ((i == rl) ? '+' : '-');
                        }
                        // Output the predicted label.
                        os << ((i == inst.argmax()) ? '+' : '-');
                        os << labels[i];

                        // Output the score/probability if necessary.
                        if (opt.output & option::OUTPUT_PROBABILITY) {
                            os << opt.value_separator << inst.prob(i);
                        } else if (opt.output & option::OUTPUT_SCORE) {
                            os << opt.value_separator << inst.score(i);
                        }

                        os << std::endl;
                        os << comments[i];
                    }

                } else {
                    // Output the reference label.
                    if (opt.output & option::OUTPUT_RLABEL) {
                        os << labels[rl] << opt.token_separator;
                    }
                    // Output the predicted label.
                    os << labels[inst.argmax()];

                    // Output the score/probability if necessary.
                    if (opt.output & option::OUTPUT_PROBABILITY) {
                        os << opt.value_separator << inst.prob(inst.argmax());
                    } else if (opt.output & option::OUTPUT_SCORE) {
                        os << opt.value_separator << inst.score(inst.argmax());
                    }

                    os << std::endl;

                }

                // Output EOI.
                os << "@eoi" << std::endl;
            }

            // Accumulate the performance.
            if (opt.test) {
                acc.set(inst.argmax() == rl);
            }

            rl = -1;
            inst.clear();
            labels.clear();
            comments.clear();
            inner = false;

        } else {
            std::string label;
            bool truth = false;
            parse_line(inst, fgen, label, truth, opt, line, lines);
            if (truth) {
                rl = inst.size() - 1;
            }
            labels.push_back(label);
            if (labels.size() != inst.size()) {
                throw invalid_data("", line, lines);
            }
            if ((int)comments.size() < inst.size()) {
                comments.resize(inst.size());
            }
        }
    }

    // Output the performance if necessary.
    if (opt.test) {
        acc.output(os);
    }

    return 0;
}
