/*
 *		Utilities for training.
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

#ifndef __TRAIN_H__
#define __TRAIN_H__

#include <algorithm>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>
#include <libexecstream/exec-stream.h>
#include <util.h>

template <
    class trainer_type,
    class data_type>
static void
set_parameters(
    trainer_type& trainer,
    data_type& data,
    const option& opt
    )
{
    typename option::params_type::const_iterator itp;
    classias::parameter_exchange& params = trainer.params();
    for (itp = opt.params.begin();itp != opt.params.end();++itp) {
        std::string name, value;
        std::string::size_type pos = itp->find('=');
        if (pos != itp->npos) {
            name = std::string(*itp, 0, pos);
            value = itp->substr(pos+1);
        } else {
            name = *itp;
        }
        params.set(name, value);
    }
}

template <class data_type>
static int
split_data(
    data_type& data,
    const option& opt
    )
{
    int i = 0;
    typename data_type::iterator it;
    for (it = data.begin();it != data.end();++it, ++i) {
        it->set_group(i % opt.split);
    }
    return opt.split;
}

template <class data_type>
static void
read_data(
    data_type& data,
    const option& opt
    )
{
    std::ostream& os = std::cout;
    std::ostream& es = std::cerr;

    // Read files for training data.
    if (opt.files.empty()) {
        // Read the data from STDIN.
        os << "STDIN" << std::endl;
        read_stream(std::cin, data, opt, 0);
    } else {
        // Read the data from files.
        for (int i = 0;i < (int)opt.files.size();++i) {
            std::string decomp, decomp_cmd, decomp_arg;
            const std::string& file = opt.files[i];

            // Set a compressor and its arguments.
            if (file.compare(file.length()-3, 3, ".gz") == 0) {
                decomp = " (gzip)";
                decomp_cmd = "gzip";
                decomp_arg = "-dc";
            } else if (file.compare(file.length()-4, 4, ".bz2") == 0) {
                decomp = " (bzip2)";
                decomp_cmd = "bzip2";
                decomp_arg = "-dck";
            } else if (file.compare(file.length()-3, 3, ".xz") == 0) {
                decomp = " (xz)";
                decomp_cmd = "xz";
                decomp_arg = "-dck";
            }

            // Output the file name (and its decompressor).
            os << "- " << i+1 << decomp << ": " << file;
            os.flush();

            if (decomp_cmd.empty()) {
                // Read an uncompressed file.
                std::ifstream ifs(file.c_str());
                if (!ifs.fail()) {
                    read_stream(ifs, data, opt, i);
                } else {
                    os << ": failed" << std::endl;
                    throw invalid_data("An error occurred when reading a file", 0);
                }
            } else {
                // Read a compressed file from an external decompressor.
                exec_stream_t proc;
                proc.set_text_mode(exec_stream_t::s_out);
                proc.start(decomp_cmd, decomp_arg.c_str(), file.c_str());
                std::istream& ifs = proc.out();
                if (!ifs.fail()) {
                    read_stream(ifs, data, opt, i);
                    proc.close();
                    if (proc.exit_code() != 0) {
                        os << ": (exit_code = " << proc.exit_code() << ")";
                    }
                } else {
                    os << ": failed (" << proc.exit_code() << ")" << std::endl;
                    throw invalid_data("An error occurred when decompressing a file", 0);
                }
            }
            os << std::endl;
            os.flush();
        }
    }
}

template <class data_type>
static int
read_dataset(
    data_type& data,
    const option& opt
    )
{
    // Read the training data.
    read_data(data, opt);

    // Finalize the data.
    finalize_data(data, opt);

    // Shuffle instances if necessary.
    if (opt.shuffle) {
        std::random_shuffle(data.begin(), data.end());
    }

    // Split the training data if necessary.
    if (0 < opt.split) {
        split_data(data, opt);
        return opt.split;
    } else {
        return (int)opt.files.size();
    }
}

template <
    class data_type,
    class trainer_type
>
static int
train(option& opt)
{
    stopwatch sw;
    data_type data;
    int num_groups = 0;
    std::ostream& os = opt.os;

    // Show the help message for the algorithm and exit if necessary.
    if (opt.mode == option::MODE_HELP_ALGORITHM) {
        trainer_type tr;
        tr.params().help(opt.os);
        return 0;
    }

	// Report the start time and global configurations.
    os << "Task type: ";
    switch (opt.type) {
    case option::TYPE_BINARY:       os << "binary";         break;
    case option::TYPE_MULTI_DENSE:  os << "multi-dense";    break;
    case option::TYPE_MULTI_SPARSE: os << "multi-sparse";   break;
    case option::TYPE_CANDIDATE:    os << "candidate";      break;
    }
    os << std::endl;
    os << "Training algorithm: " << opt.algorithm << std::endl;
    os << "Instance shuffle: " << std::boolalpha << opt.shuffle << std::endl;
    os << "Bias feature generation: " << std::boolalpha << opt.generate_bias << std::endl;
    os << "Model file: " << opt.model << std::endl;
    os << "Instance splitting: " << opt.split << std::endl;
    os << "Holdout group: " << opt.holdout << std::endl;
    os << "Cross validation: " << std::boolalpha << opt.cross_validation << std::endl;
    os << "Attribute filter: " << opt.filter_string << std::endl;
    os << "Start time: " << timestamp << std::endl;
    os << std::endl;

    // Read the source data.
    os << "Reading the data set from " << opt.files.size() << " files" << std::endl;
    sw.start();
    num_groups = read_dataset(data, opt);
    sw.stop();
    os << "Number of instances: " << data.size() << std::endl;
    os << "Number of groups: " << num_groups << std::endl;
    os << "Number of attributes: " << data.num_attributes() << std::endl;
    os << "Number of labels: " << data.num_labels() << std::endl;
    os << "Number of features: " << data.num_features() << std::endl;
    os << "Seconds required: " << sw.get() << std::endl;
    os << std::endl;

    // Exit if the data set is empty.
    if (data.empty()) {
        throw invalid_data("The data set is empty", 0);
    }

    // Start training.
    if (opt.cross_validation) {
        // Training with cross validation
        for (int i = 0;i < num_groups;++i) {
            // Set training parameters.
            trainer_type trainer;
            set_parameters(trainer, data, opt);

            os << "===== Cross validation (" << (i + 1) << "/" << num_groups << ") =====" << std::endl;
            sw.start();
            trainer.train(data, opt.os, i);
            sw.stop();
            os << "Seconds required: " << sw.get() << std::endl;
            os << std::endl;
        }
    } else {
        // Set training parameters.
        trainer_type trainer;
        set_parameters(trainer, data, opt);

        // Start training.
        sw.start();
        trainer.train(data, opt.os, (0 < opt.holdout ? (opt.holdout-1) : -1));
        sw.stop();
        os << "Seconds required: " << sw.get() << std::endl;
        os << std::endl;

        // Store the model.
        if (!opt.model.empty()) {
            output_model(data, trainer.model(), opt);
        }
    }

	// Report the finish time.
    os << "Finish time: " << timestamp << std::endl;
    os << std::endl;

    return 0;
}

#endif/*__TRAIN_H__*/
