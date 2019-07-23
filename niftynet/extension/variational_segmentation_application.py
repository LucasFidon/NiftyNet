import tensorflow as tf
# from niftynet.application.base_application import BaseApplication
from niftynet.application.segmentation_application import SegmentationApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
# from niftynet.engine.sampler_grid_v2 import GridSampler
# from niftynet.engine.sampler_resize_v2 import ResizeSampler
# from niftynet.engine.sampler_uniform_v2 import UniformSampler
# from niftynet.engine.sampler_weighted_v2 import WeightedSampler
# from niftynet.engine.sampler_balanced_v2 import BalancedSampler
# from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
# from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
# from niftynet.io.image_reader import ImageReader
# from niftynet.layer.binary_masking import BinaryMaskingLayer
# from niftynet.layer.discrete_label_normalisation import \
#     DiscreteLabelNormalisationLayer
# from niftynet.layer.histogram_normalisation import \
#     HistogramNormalisationLayer
from niftynet.layer.loss_segmentation import LossFunction
# from niftynet.layer.mean_variance_normalisation import \
#     MeanVarNormalisationLayer
# from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
# from niftynet.layer.rand_flip import RandomFlipLayer
# from niftynet.layer.rand_rotation import RandomRotationLayer
# from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.evaluation.segmentation_evaluator import SegmentationEvaluator
# from niftynet.layer.rand_elastic_deform import RandomElasticDeformationLayer
import pandas as pd
from itertools import chain, combinations
import random
import numpy as np


SUPPORTED_INPUT = set(['image', 'label', 'weight', 'sampler', 'inferred'])
MODALITIES_img = ['T1', 'T1c', 'T2', 'Flair']

np.random.seed(0)
tf.set_random_seed(1)

def all_subsets(l):
    #Does not include the empty set and l
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES_img)


def KL_divergence(mu_1, logvar_1, mu_2, logvar_2):
    " KLD(p_1 || p_2)"
    var_1 = tf.exp(logvar_1)
    var_2 = tf.exp(logvar_2)
    KLD = 1/2*tf.reduce_mean(-1  + logvar_2 - logvar_1 + (var_1+tf.square(mu_1-mu_2))/(var_2+1e-7))
    return KLD

def Product_Gaussian(means, logvars, list_mod):
    mu_prior = tf.zeros(tf.shape(means[list_mod[0]]))
    log_prior = tf.zeros(tf.shape(means[list_mod[0]]))

    eps=1e-7
    T = [1/(tf.exp(logvars[mod]) + eps)  for mod in list_mod]  + [1+log_prior]
    mu = [means[mod]/(tf.exp(logvars[mod]) + eps) for mod in list_mod] + [mu_prior]

    posterior_means = tf.add_n(mu) / tf.add_n(T)
    var = 1 / tf.add_n(T)
    posterior_logvars = tf.log(var + eps)

    return posterior_means, posterior_logvars

def Product_Gaussian_main(means, logvars, list_mod, choices):
    mu_prior = tf.zeros(tf.shape(means[list_mod[0]]))
    log_prior = tf.zeros(tf.shape(means[list_mod[0]]))


    eps=1e-7
    T = tf.boolean_mask([1/(tf.exp(logvars[mod]) + eps)  for mod in list_mod],   choices)
    mu = tf.boolean_mask([means[mod]/(tf.exp(logvars[mod]) + eps) for mod in list_mod],  choices)

    T = tf.concat([T,[1+log_prior]], 0)
    mu = tf.concat([mu,[mu_prior]], 0)

    posterior_means = tf.reduce_sum(mu,0) / tf.reduce_sum(T,0)
    var = 1 / tf.reduce_sum(T,0)
    posterior_logvars = tf.log(var + eps)

    return posterior_means, posterior_logvars

def compute_KLD(means,logvars, choices):
    # Prior parameters
    mu_prior = tf.zeros(tf.shape(means[MODALITIES_img[0]]))
    log_prior = tf.zeros(tf.shape(means[MODALITIES_img[0]]))

    # Full modalities
    full_means, full_logvars = Product_Gaussian_main(means, logvars, MODALITIES_img, choices)

    # Initialization sums
    sum_inter_KLD = 0
    sum_prior_KLD = 0

    sum_prior_KLD += KL_divergence(full_means, full_logvars, mu_prior, log_prior)

    return 0, sum_prior_KLD


class VariationalSegmentationApplication(SegmentationApplication):
    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, action):
        super(SegmentationApplication, self).__init__()
        tf.logging.info('starting segmentation application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.segmentation_param = None
        self.SUPPORTED_SAMPLING = {
            'uniform': (self.initialise_uniform_sampler,
                        self.initialise_grid_sampler,
                        self.initialise_grid_aggregator),
            'weighted': (self.initialise_weighted_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
            'resize': (self.initialise_resize_sampler,
                       self.initialise_resize_sampler,
                       self.initialise_resize_aggregator),
            'balanced': (self.initialise_balanced_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
        }

    def set_iteration_update(self, iteration_message):
        """
        This function will be called by the application engine at each
        iteration.
        """
        # current_iter = iteration_message.current_iter
        # if iteration_message.is_training:
        #     if current_iter % 2 == 0:
        #         iteration_message.data_feed_dict[self.is_brats] = True
        #     else:
        #         iteration_message.data_feed_dict[self.is_brats] = False

        # nb_choices = np.random.randint(4)
        # choices = np.random.choice(4, nb_choices+1, replace=False, p=[1/4,1/4,1/4,1/4])
        # choices = [True if k in choices else False for k in range(4)]
        choices = [True] * 4  # use all the modalities all the time
        print(choices)
        iteration_message.data_feed_dict[self.choices] = choices

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        # def data_net(for_training):
        #    with tf.name_scope('train' if for_training else 'validation'):
        #        sampler = self.get_sampler()[0][0 if for_training else -1]
        #        data_dict = sampler.pop_batch_op()
        #        image = tf.cast(data_dict['image'], tf.float32)
        #        return data_dict, self.net(image, is_training=for_training)

        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()


        self.var = tf.placeholder_with_default(0, [], 'var')
        self.choices = tf.placeholder_with_default([True, True, True, True], [4], 'choices')
        # choose_all will be used when we want to use all the modalities at all each training iteration
        self.choose_all = tf.fill([4], True, name='default_choice')

        if self.is_training:
            # if self.action_param.validation_every_n > 0:
            #    data_dict, net_out = tf.cond(tf.logical_not(self.is_validation),
            #                                 lambda: data_net(True),
            #                                 lambda: data_net(False))
            # else:
            #    data_dict, net_out = data_net(True)
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            image_unstack = tf.unstack (image, axis=-1)
            print('start training...')
            print(image)
            # net_img, post_param = self.net({MODALITIES_img[k]: tf.expand_dims(image_unstack[k],-1) for k in range(4)},
            #                                self.choices, is_training=self.is_training)
            # No missing modalities
            net_img, post_param = self.net({MODALITIES_img[k]: tf.expand_dims(image_unstack[k], -1) for k in range(4)},
                                           self.choose_all, is_training=self.is_training)

            net_seg = net_img['seg']
            net_img = tf.concat([net_img[mod] for mod in MODALITIES_img],axis=-1)
            

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
             
            print('seeg')
            gt =  data_dict['label']

            cross = LossFunction(
                n_class=4,
                loss_type='CrossEntropy')

            dice = LossFunction(
                n_class=4,
                loss_type='Dice',
                softmax=True)

            wasserstein_dice = LossFunction(
                n_class=4,
                loss_type='WGDL'
            )

            #TODO: try with Focal Loss instead of Cross Entropy


            gt =  data_dict['label']
            loss_cross = cross(prediction=net_seg,ground_truth=gt, weight_map=None)
            loss_dice = dice(prediction=net_seg,ground_truth=gt)
            # loss_dice = wasserstein_dice(prediction=net_seg,ground_truth=gt)
            loss_seg = loss_cross + loss_dice
            
            print('output')
            print(net_img)
            loss_reconstruction = tf.reduce_mean(tf.square(net_img - image))


            print('output_seg')
            print(net_seg)

            print('gt')
            print(gt)

            sum_inter_KLD = 0.0
            sum_prior_KLD = 0.0
            nb_skip = len(post_param)
            for k in range(nb_skip):
                inter_KLD, prior_KLD = compute_KLD(post_param[k]['mu'], post_param[k]['logvar'], self.choices)
                sum_inter_KLD += inter_KLD
                sum_prior_KLD += prior_KLD

            KLD = 1/nb_skip*sum_inter_KLD + 1/nb_skip*sum_prior_KLD

            data_loss =  loss_seg + 0.1*KLD + 0.1*loss_reconstruction

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = tf.reduce_mean(
                [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
            loss = data_loss + reg_loss

            grads = self.optimiser.compute_gradients(loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=loss, name='loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=loss, name='loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=KLD, name='KLD',
                average_over_devices=False, collection=CONSOLE)   
            outputs_collector.add_to_collection(
                var=loss_reconstruction, name='loss_reconstruction',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.choices, name='choices',
                average_over_devices=False, collection=CONSOLE)                  
  

        elif self.is_inference:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            print('start inference...')
            print(image)
            image = tf.unstack (image, axis=-1)
            # inference only with all modalities
            net_img, post_param = self.net(
                {MODALITIES_img[k]: tf.expand_dims(image[k],-1) for k in range(4)},
                [True,True,True,True],
                is_training=self.is_training,
                is_inference=True)

            net_seg = net_img['seg']

            print('output')
            post_process_layer = PostProcessingLayer(
                    'ARGMAX', num_classes=4)
            net_seg = post_process_layer(net_seg)
            outputs_collector.add_to_collection(
                var=net_img['T1'], name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            self.initialise_aggregator()

    def interpret_output(self, batch_output):
        if self.is_inference:
            print(self.var.eval())
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        return True

    def initialise_evaluator(self, eval_param):
        self.eval_param = eval_param
        self.evaluator = SegmentationEvaluator(self.readers[0],
                                               self.segmentation_param,
                                               eval_param)

    def add_inferred_output(self, data_param, task_param):
        return self.add_inferred_output_like(data_param, task_param, 'label')
