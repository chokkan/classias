/*
 *		A sample program for training a binary classifier with online setting.
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

#include <iostream>
#include <string>
#include <vector>

#include <classias/classias.h>
#include <classias/classify/linear/binary.h>
#include <classias/train/pegasos.h>

#include "strsplit.h"   // necessary for strsplit() and get_id_value().

typedef classias::expandable_weight_vector model_type;

// Define the type of a training algorithm. Change this type to use a
// different online training algorithm.
typedef classias::train::pegasos_binary<
    classias::classify::linear_binary_hinge<model_type>
    >
    trainer_type;

int main(int argc, char *argv[])
{
    trainer_type tr;
    std::istream& is = std::cin;
    std::ostream& os = std::cout;
    std::ostream& es = std::cerr;

    // Show the algorithm name and parameters.
    tr.copyright(es);
    tr.params().show(es);
    es << std::endl;

    // Initialize the trainer.
    tr.start();

    // Read a data set from STDIN.
    for (;;) {
        // Read a line.
        std::string line;
        std::getline(is, line);
        if (is.eof()) {
            break;
        }

        // Split the line into fields with space characters.
        std::vector<std::string> fields;
        strsplit(fields, line);

        // The line must have at least a label and a feature.
        if (fields.size() > 2) {
            // An instance object.
            classias::binstance inst;
            // Set the boolean label for the instance.
            inst.set_label(fields[0] != "-1");

            // Loop over the rest of fields.
            for (size_t i = 1;i < fields.size();++i) {
                // Split the field into a feature identifier and value.
                int fid;
                double value;
                get_id_value(fields[i], fid, value, ':');
                
                // Append the feature to the instance.
                inst.append(fid, value);
            }

            // Update the model by using the current instance.
            tr.update(&inst);
        }
    }

    // Pause the training process, and report the progress.
    tr.discontinue();
    tr.report(es);

    // Finalize the trainer.
    tr.finish();

    // Output the model.
    const model_type& w = tr.model();
    for (size_t i = 0;i < w.size();++i) {
        // Feature ID and its weight.
        os << i << '\t' << w[i] << std::endl;
    }

    return 0;
}
