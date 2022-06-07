import re
import tensorflow as tf
tf.set_random_seed(0)
import numpy as np

from cmain import preset_args, args_init
from cEADQN import DeepQLearner
from cEnvironment import Environment
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics.pairwise import cosine_similarity

class Agent(object):
    """docstring for Agent"""

    def __init__(self, args, sess):

        self.env_act = Environment(args, 'act')
        self.net_act = DeepQLearner(args, 'act', 'channels_last')


        self.env_arg = Environment(args, 'arg')
        self.net_arg = DeepQLearner(args, 'arg', 'channels_last')
        self.num_words = args.num_words
        self.context_len = 128

    def predict(self, text):
        # e.g. text = ['Cook the rice the day before.', 'Use leftover rice.']
        sentence_emb_list = self.env_act.init_predict_act_text(text)

        sentence_emb_list = np.stack(sentence_emb_list)

        # sentence_emb_list /= np.linalg.norm(sentence_emb_list, axis=1, keepdims=True)

        # cos_sim =cosine_similarity(sentence_emb_list, Y=None, dense_output=True)
        # print(cos_sim)
        # np.save("cos_sim_%s"%filename,cos_sim)
        # Generate a mask for the upper triangle

        sents = []  # dictionary for last sentence, this sentence and actions
        for i in range(len(self.env_act.current_text['sents'])):
            last_sent = self.env_act.current_text['sents'][i - 1] if i > 0 else []
            this_sent = self.env_act.current_text['sents'][i]
            sents.append({'last_sent': last_sent, 'this_sent': this_sent, 'acts': []})

        for i in range(self.num_words):
            state_act = self.env_act.getState()
            qvalues_act = self.net_act.predict(state_act)
            # print(qvalues_act)
            action_act = np.argmax(qvalues_act[0])
            # print(action_act)
            self.env_act.act_online(action_act, i)
            if action_act == 1:
                last_sent, this_sent = self.env_arg.init_predict_arg_text(i, self.env_act.current_text)
                for j in range(self.context_len):
                    state_arg = self.env_arg.getState()
                    qvalues_arg = self.net_arg.predict(state_arg)
                    # print(qvalues_arg)
                    action_arg = np.argmax(qvalues_arg[0])
                    # print(action_arg)
                    self.env_arg.act_online(action_arg, j)
                    if self.env_arg.terminal_flag:
                        break

                act_idx = i
                obj_idxs = []
                sent_words = self.env_arg.current_text['tokens']
                tmp_num = self.context_len if len(sent_words) >= self.context_len else len(sent_words)

                for j in range(tmp_num): #until context_len or word_len
                    if self.env_arg.state[j, -1] == 2:
                        obj_idxs.append(j) #append j if state's last value is 2.
                        if j == len(sent_words) - 1: # if last word, reset j to -1
                            j = -1
                # if len(obj_idxs) == 0:
                #     obj_idxs.append(-1) #if there are no object indexes, append UNK
                si, ai = self.env_act.current_text['word2sent'][i]
                ai += len(sents[si]['last_sent'])
                sents[si]['acts'].append({'act_idx': ai, 'obj_idxs': [obj_idxs, []],
                                          'act_type': 1, 'related_acts': []})
            if self.env_act.terminal_flag:
                break
        return sents

