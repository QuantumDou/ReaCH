import torch
import torch.nn.functional as F
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from numpy import inf
import re
from string import punctuation


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(),"CIDER")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res 




def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # /users/k1623928/.conda/envs/tian_env/lib/python3.8/site-packages 的
        # torch.use_deterministic_algorithms(False)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # torch.use_deterministic_algorithms(True)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def get_report(decoded_sequences):
    if 'so the report is ' in decoded_sequences:
        parts = decoded_sequences.split(" so the report is ")[1:]
        cut_decoded_sequences = " ".join(parts[:]).strip()
    else:
        cut_decoded_sequences = " ".join(decoded_sequences.split()[0:])
    return cut_decoded_sequences


class Recorder():
    def __init__(self, mode='test'):
        self.mode = mode
        self.best_epoch = 0
        self.best_score = -inf  
        self.best_recorder = {
            self.mode: {
                f'{self.mode}_BLEU_4': float('-inf'),
            }
        }

    def record_best(self,log):
        update_flag = (log[f'{self.mode}_BLEU_4'] >= self.best_recorder[self.mode][f'{self.mode}_BLEU_4']) 
        if update_flag:
                self.best_recorder[self.mode].update(log)

    
    def needsUpdate(self,log,epoch):
        needs_update = (log[f'{self.mode}_BLEU_4'] >= self.best_score)
        if needs_update:
            self.best_score = log[f'{self.mode}_BLEU_4'] 
            self.best_epoch = epoch
        return needs_update

    def check_early_stop(self,current_epoch,patience=5):
        return (current_epoch - self.best_epoch) >= patience


def parse_decimal(text):
    find_float = lambda x: re.search("\d+(\.\d+)", x).group()
    text_list = []
    for word in text.split():
        try:
            decimal = find_float(word)
            new_decimal = decimal.replace(".", "*")
            text_list.append(new_decimal)
        except:
            text_list.append(word)
    return " ".join(text_list)

def clean_train_sentence(text):
    punc = list(punctuation)
    text = re.sub(r"xxxx", " ", text)
    text = re.sub("[^a-z\s]", "", text.lower())
    text_nopunc = [char for char in text if char not in punc] 
    text_nopunc = "".join(text_nopunc)
    wd = []
    for word in text_nopunc.split(): 
        wd.append(word)
    sentence = " ".join(wd)
    if sentence.strip()=='images':
        return []
    return sentence

def clean_train_report(report):
        report = parse_decimal(report)
        report_cleaner = lambda t: t.replace('. .', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('.')
        
        tokens = [clean_train_sentence(sent) for sent in report_cleaner(report) if sent!='' if clean_train_sentence(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report


def clean_iu_test_report(report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

def clean_mimic_test_report(report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report


