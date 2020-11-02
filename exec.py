#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""execution script."""

import argparse
import os
import time
import torch

import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor
from plotting import plot_batch_prediction
import detector
import data_loader
import configs as cf
import logging

def train(logger, model, ckpt_pth):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics, TrainingPlot = utils.prepare_monitoring(cf)

    if ckpt_pth is not None:
        starting_epoch, _monitor_metrics = utils.load_checkpoint(ckpt_pth, model, optimizer)
        logger.info('resumed to checkpoint {} at epoch {}'.format(ckpt_pth, starting_epoch))
        for i in range(len(_monitor_metrics['train']['monitor_values'])):
            monitor_metrics['train']['monitor_values'][i] = _monitor_metrics['train']['monitor_values'][i]
            monitor_metrics['val']['monitor_values'][i] = _monitor_metrics['val']['monitor_values'][i]
        monitor_metrics['train']['nodule_ap'] = _monitor_metrics['train']['nodule_ap']
        monitor_metrics['val']['nodule_ap'] = _monitor_metrics['val']['nodule_ap']

    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)
    is_begining = True

    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        start_time = time.time()

        model.train()
        train_results_list = []

        for bix in range(cf.num_train_batches):
            batch = next(batch_gen['train'])
            tic_fw = time.time()
            results_dict = model.train_forward(batch)
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            logger.info('tr. batch {0}/{1} (ep. {2}) fw {3:.3f}s / bw {4:.3f}s / total {5:.3f}s || '
                        .format(bix + 1, cf.num_train_batches, epoch, tic_bw - tic_fw,
                                time.time() - tic_bw, time.time() - tic_fw) + results_dict['logger_string'])
            train_results_list.append([results_dict['boxes'], batch['pid']])
            monitor_metrics['train']['monitor_values'][epoch].append(results_dict['monitor_values'])

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])
        train_time = time.time() - start_time

        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        with torch.no_grad():
            model.eval()
            if cf.do_validation:
                val_results_list = []
                val_predictor = Predictor(cf, model, logger, mode='val')
                for _ in range(batch_gen['n_val']):
                    batch = next(batch_gen[cf.val_mode])
                    if cf.val_mode == 'val_patient':
                        results_dict = val_predictor.predict_patient(batch)
                    elif cf.val_mode == 'val_sampling':
                        results_dict = model.train_forward(batch, is_validation=True)
                    val_results_list.append([results_dict['boxes'], batch['pid']])
                    monitor_metrics['val']['monitor_values'][epoch].append(results_dict['monitor_values'])

                _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
                model_selector.run_model_selection(model, optimizer, monitor_metrics, epoch)

            # update monitoring and prediction plots
            TrainingPlot.update_and_save(monitor_metrics, epoch, is_begining = is_begining)
            is_begining = False
            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(
                epoch, epoch_time, train_time, epoch_time-train_time))


def test(logger, model, ckpt_pth):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    model = model.cuda()
    test_predictor = Predictor(cf, model, logger, mode='test')
    batch_gen = data_loader.get_test_generator(cf, logger)
    test_results_list = test_predictor.predict_test_set(batch_gen, return_results=True)
    return test_results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,  default='train', help='one out of: train / test')
    parser.add_argument('-pred_var', default='False')
    parser.add_argument('-mc_var', default='False')
    parser.add_argument('--ckpt_pth', type=str, default=None)
    parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")

    args = parser.parse_args()
    fold = args.fold
    ckpt_pth = args.ckpt_pth


    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if (not args.pred_var) and (not args.mc_var):
        exp_path = 'exp_path/baseline'
    elif args.mc_var and (not args.pred_var):
        exp_path = 'exp_path/mc_var'
    elif args.mc_var and args.pred_var:
        exp_path = 'exp_path/both_var'
    else:
        exp_path = 'exp_path/tmp'
    
    cf.fold_dir = exp_path
    log_file = exp_path + '/exec.log'
    hdlr = logging.FileHandler(log_file)
    print('Logging to {}'.format(log_file))
    logger.addHandler(hdlr)
    logger.propagate = False

    if args.dev:
        cf.batch_size, cf.num_epochs, cf.min_save_thresh, cf.save_n_models = 3 
        cf.num_train_batches, cf.num_val_batches, cf.max_val_patients = 5, 1, 1
        cf.test_n_epochs =  1
        cf.max_test_patients = 1

    model = detector.net(cf, logger=logger, mc_var=args.mc_var, pred_var=args.pred_var)
    
    if args.mode == 'train':
        train(logger, model, ckpt_pth)

    if args.mode == 'test':
        test(logger, model, ckpt_pth, args.mc_var)

    for hdlr in logger.handlers:
        hdlr.close()
